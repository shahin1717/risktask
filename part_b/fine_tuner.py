import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part_a'))

import torch
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from jiwer import wer

from data_loader import loader
from dataset import build_dataset, build_collator


# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR    = "../part_a/data"
MODEL_NAME  = "openai/whisper-small"
OUTPUT_DIR  = "./whisper-az-finetuned"
TRAIN_SAMPLES = 150
VAL_SAMPLES   = 30
EPOCHS        = 10
BATCH_SIZE    = 8
LEARNING_RATE = 1e-5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ── Metric ───────────────────────────────────────────────────────────────────

def compute_metrics(processor: WhisperProcessor):
    """
    Returns a metric function compatible with Seq2SeqTrainer.
    Computes WER on validation set each epoch.
    """
    def _compute(eval_preds):
        pred_ids, label_ids = eval_preds

        # Replace -100 (ignored padding) back to pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and references
        predictions = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        references  = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        score = wer(references, predictions)

        return {"wer": round(score, 4)}

    return _compute


# ── Loader ───────────────────────────────────────────────────────────────────

def load_data(processor: WhisperProcessor):
    """
    Load and prepare train and validation datasets.

    Returns:
        train_dataset, val_dataset
    """
    print("Loading datasets...")
    raw_train = loader(DATA_DIR, split="train")
    raw_val   = loader(DATA_DIR, split="validation")

    train_dataset = build_dataset(raw_train, processor, num_samples=TRAIN_SAMPLES)
    val_dataset   = build_dataset(raw_val,   processor, num_samples=VAL_SAMPLES)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")

    return train_dataset, val_dataset


# ── Trainer ──────────────────────────────────────────────────────────────────

def build_trainer(model, processor, train_dataset, val_dataset):
    """
    Build Seq2SeqTrainer with all training arguments.

    Returns:
        Seq2SeqTrainer
    """
    args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        num_train_epochs            = EPOCHS,
        learning_rate               = LEARNING_RATE,

        # Evaluation + saving
        eval_strategy       = "epoch",
        save_strategy       = "epoch",
        load_best_model_at_end      = True,        # saves best checkpoint by WER
        metric_for_best_model       = "wer",
        greater_is_better           = False,       # lower WER is better

        # Whisper specific
        predict_with_generate       = True,
        generation_max_length       = 225,

        # Logging
        logging_dir                 = "./logs",
        logging_strategy            = "epoch",
        report_to                   = "none",      # disable wandb

        # GPU
        fp16                        = torch.cuda.is_available(),
    )

    collator = build_collator(processor)

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = args,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        processing_class = processor.feature_extractor,
        data_collator    = collator,
        compute_metrics  = compute_metrics(processor),
    )

    return trainer


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}")

    # Load processor and model
    print(f"Loading model: {MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="azerbaijani", task="transcribe")
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    # Freeze encoder — only fine-tune decoder (faster + less overfitting)
    model.freeze_encoder()

    # Load data
    train_dataset, val_dataset = load_data(processor)

    # Build trainer
    trainer = build_trainer(model, processor, train_dataset, val_dataset)

    # Train
    print("\nStarting fine-tuning...")
    trainer.train()

    # Save best model
    print(f"\nSaving best model to {OUTPUT_DIR}/best")
    trainer.save_model(f"{OUTPUT_DIR}/best")
    processor.save_pretrained(f"{OUTPUT_DIR}/best")

    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()