"""QLoRA fine-tune of Qwen2.5-7B-Instruct on CORD-v2 field extraction.

Same shape as Project 1's `train_extractor.py`: a `Config` object and a `run_training`
that returns its metrics, so `sweep.py` can drive a grid without relaunching the
interpreter or re-reading files.

QLoRA rather than full fine-tuning is the point of the project, not a compromise. The
base weights are frozen in 4-bit NF4 and only the adapter trains, which is what makes a
7B fine-tune fit on one A100 at all - and it is also what makes the eventual efficiency
comparison against a 125M encoder honest rather than rhetorical.

    python -m src.train_qlora --max-train 40 --epochs 1     # smoke test first
    python -m src.train_qlora --epochs 3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

from . import manifest
from .data import DATASET_ID
from .eval_llm import MODEL_ID, evaluate, hf_generate_fn
from .sft_data import build_sft_dataset

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Every linear layer in the attention and MLP blocks. Restricting LoRA to the attention
# projections trains fewer parameters but consistently scores lower on structured output
# tasks, where the MLPs carry a lot of the formatting behaviour.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Config:
    """Plain holder so run_training takes one object whether called by CLI or sweep."""

    def __init__(
        self,
        epochs: float = 3.0,
        lr: float = 2e-4,
        batch_size: int = 4,
        grad_accum: int = 4,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        max_length: int = 4096,
        seed: int = 0,
        model: str = MODEL_ID,
        output: str = "outputs/qwen-cord-lora",
        max_train: int = 0,
        run_name: str | None = None,
        write_artifacts: bool = True,
        save_model: bool = True,
        eval_limit: int = 0,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.max_length = max_length
        self.seed = seed
        self.model = model
        self.output = output
        self.max_train = max_train
        self.run_name = run_name
        self.write_artifacts = write_artifacts
        self.save_model = save_model
        self.eval_limit = eval_limit

    def as_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "lora_rank": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": TARGET_MODULES,
            "max_length": self.max_length,
            "seed": self.seed,
            "model": self.model,
            "max_train": self.max_train,
        }


def run_training(args: Config) -> dict:
    """Train one configuration and return its metrics."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    # Tracking is useful but must never be the reason a training run dies.
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        mlflow = None
        print("mlflow not installed; skipping experiment tracking")

    ARTIFACTS.mkdir(exist_ok=True)

    print(f"Loading {DATASET_ID} ...")
    train = load_dataset(DATASET_ID, split="train")
    if args.max_train:
        train = train.select(range(min(args.max_train, len(train))))
    train_ds = build_sft_dataset(train)
    print(f"{len(train_ds)} training receipts")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Double quantisation saves roughly another 0.4 bits per parameter. Free memory.
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        max_length=args.max_length,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    if mlflow:
        mlflow.set_experiment("qwen-qlora-cord-extraction")
    started = time.time()
    with (mlflow.start_run(run_name=args.run_name) if mlflow else contextlib.nullcontext()):
        if mlflow:
            mlflow.log_params({**args.as_dict(), "train_size": len(train_ds)})

        result = trainer.train()

        metrics = {
            **args.as_dict(),
            "train_size": len(train_ds),
            "train_runtime_seconds": round(result.metrics["train_runtime"], 1),
            "final_train_loss": round(result.metrics["train_loss"], 4),
            "trainable_parameters": sum(
                p.numel() for p in trainer.model.parameters() if p.requires_grad
            ),
            "total_parameters": sum(p.numel() for p in trainer.model.parameters()),
        }

        if args.save_model:
            trainer.save_model(args.output)
            tokenizer.save_pretrained(args.output)
            metrics["adapter_path"] = args.output
            metrics["adapter_bytes"] = sum(
                path.stat().st_size for path in Path(args.output).rglob("*") if path.is_file()
            )

        # Scored with the model still in memory: reloading 7B parameters to grade them is
        # minutes per sweep configuration for no gain.
        if args.eval_limit:
            validation = load_dataset(DATASET_ID, split="validation").select(
                range(args.eval_limit)
            )
            model.config.use_cache = True
            trainer.model.eval()
            scores = evaluate(
                validation,
                hf_generate_fn(trainer.model, tokenizer),
                config=args.run_name or "qlora-validation",
            )
            metrics.update(
                {
                    "validation_f1": scores["test_f1"],
                    "validation_parse_rate": scores["parse_rate"],
                    "validation_locate_rate": scores["locate_rate"],
                }
            )

        if mlflow:
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )

    if args.write_artifacts:
        (ARTIFACTS / "training_config.json").write_text(json.dumps(metrics, indent=2))
        manifest.write(
            "qlora_training",
            dataset=DATASET_ID,
            split="train",
            quantization="nf4 double-quant, bf16 compute",
            **args.as_dict(),
        )

    metrics["wall_clock_seconds"] = round(time.time() - started, 1)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/qwen-cord-lora")
    parser.add_argument("--max-train", type=int, default=0, help="0 = use all")
    parser.add_argument("--eval-limit", type=int, default=0, help="validation docs to score")
    cli = parser.parse_args()
    run_training(Config(**vars(cli)))


if __name__ == "__main__":
    main()
