import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part_a'))

import torch
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data_loader import loader
from dataset import build_dataset, build_collator


# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR      = "../part_a/data"
MODEL_NAME    = "openai/whisper-small"
OUTPUT_DIR    = "./whisper-az-finetuned"
TRAIN_SAMPLES = 150
VAL_SAMPLES   = 30
EPOCHS        = 10
BATCH_SIZE    = 8
LEARNING_RATE = 1e-5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ── Metric ───────────────────────────────────────────────────────────────────

wer_metric = evaluate.load("wer")

def compute_metrics(processor):
    def _compute(eval_preds):
        pred_ids, label_ids = eval_preds

        # Replace -100 back to pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode
        predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
        references  = processor.batch_decode(label_ids, skip_special_tokens=True)

        # WER (multiplied by 100 for percentage)
        wer = 100 * wer_metric.compute(predictions=predictions, references=references)
        return {"wer": round(wer, 2)}

    return _compute


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(processor):
    print("Loading and pre-processing datasets...")

    raw_train = loader(DATA_DIR, split="train")
    raw_val   = loader(DATA_DIR, split="validation")

    train_dataset = build_dataset(raw_train, processor, num_samples=TRAIN_SAMPLES)
    val_dataset   = build_dataset(raw_val,   processor, num_samples=VAL_SAMPLES)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")

    return train_dataset, val_dataset


# ── Trainer ──────────────────────────────────────────────────────────────────

def build_trainer(model, processor, train_dataset, val_dataset):

    args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        num_train_epochs            = EPOCHS,
        warmup_steps                = 50,

        # Gradient checkpointing (memory efficient on GPU)
        gradient_checkpointing      = True,
        gradient_accumulation_steps = 1,

        # Eval + save
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,

        # Whisper specific
        predict_with_generate       = True,
        generation_max_length       = 225,

        # GPU
        fp16                        = torch.cuda.is_available(),
        fp16_full_eval              = torch.cuda.is_available(),

        # Logging
        logging_strategy            = "epoch",
        report_to                   = "none",
    )

    collator = build_collator(processor)

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = collator,
        compute_metrics = compute_metrics(processor),
        processing_class= processor.feature_extractor,
    )

    return trainer


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model: {MODEL_NAME}")

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="azerbaijani", task="transcribe"
    )

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Disable cache during training (incompatible with gradient checkpointing)
    model.config.use_cache = False

    # Set generation config properly (new way — no forced_decoder_ids)
    model.generation_config.language = "azerbaijani"
    model.generation_config.task     = "transcribe"
    model.generation_config.forced_decoder_ids = None

    model.to(DEVICE)

    # Load data
    train_dataset, val_dataset = load_data(processor)

    # Build trainer and train
    trainer = build_trainer(model, processor, train_dataset, val_dataset)

    print("\nStarting fine-tuning...")
    trainer.train()

    # Save best model
    print(f"\nSaving best model to {OUTPUT_DIR}/best")
    trainer.save_model(f"{OUTPUT_DIR}/best")
    processor.save_pretrained(f"{OUTPUT_DIR}/best")

    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()