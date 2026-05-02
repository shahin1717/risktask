import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'part_a'))

import torch
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer, cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from data_loader import loader


# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR        = "../part_a/data"
BASE_MODEL      = "openai/whisper-small"
FINETUNED_MODEL = "./whisper-az-finetuned/best"
TEST_SAMPLES    = 50
SAMPLE_RATE     = 16000
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Training logs from fine_tuner.py output
TRAINING_LOGS = [
    {"epoch": 1,  "train_loss": 2.62,    "val_wer": 60.75},
    {"epoch": 2,  "train_loss": 0.8707,  "val_wer": 56.07},
    {"epoch": 3,  "train_loss": 0.581,   "val_wer": 57.48},
    {"epoch": 4,  "train_loss": 0.4075,  "val_wer": 54.21},
    {"epoch": 5,  "train_loss": 0.3001,  "val_wer": 51.87},
    {"epoch": 6,  "train_loss": 0.2366,  "val_wer": 53.27},
    {"epoch": 7,  "train_loss": 0.1937,  "val_wer": 53.74},
    {"epoch": 8,  "train_loss": 0.1659,  "val_wer": 55.61},
    {"epoch": 9,  "train_loss": 0.1501,  "val_wer": 54.67},
    {"epoch": 10, "train_loss": 0.1385,  "val_wer": 56.07},
]



# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, processor, dataset):
    """
    Run inference on dataset samples.

    Returns:
        List of dicts with reference, hypothesis, wer, cer
    """
    results = []
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="azerbaijani", task="transcribe"
    )

    model.eval()
    for i, sample in enumerate(dataset):
        try:
            audio, _ = librosa.load(sample["path"], sr=SAMPLE_RATE)

            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).input_features.to(DEVICE)

            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs,
                    forced_decoder_ids=forced_decoder_ids
                )

            hypothesis = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            reference = sample["sentence"]

            results.append({
                "reference":  reference,
                "hypothesis": hypothesis,
                "wer":        wer(reference, hypothesis),
                "cer":        cer(reference, hypothesis),
            })

            if i % 10 == 0:
                print(f"  [{i+1}/{len(dataset)}] done")

        except Exception as e:
            print(f"  Skipped {sample['path']}: {e}")

    return results


# ── Comparison Table ─────────────────────────────────────────────────────────

def compare_models(base_results, ft_results):
    """
    Print and save WER/CER comparison table.
    """
    base_wer = pd.DataFrame(base_results)["wer"].mean() * 100
    base_cer = pd.DataFrame(base_results)["cer"].mean() * 100
    ft_wer   = pd.DataFrame(ft_results)["wer"].mean()   * 100
    ft_cer   = pd.DataFrame(ft_results)["cer"].mean()   * 100

    table = pd.DataFrame([
        {"Model": "Base (whisper-small)", "WER (%)": round(base_wer, 2), "CER (%)": round(base_cer, 2)},
        {"Model": "Fine-tuned",           "WER (%)": round(ft_wer,   2), "CER (%)": round(ft_cer,   2)},
        {"Model": "Improvement",          "WER (%)": round(base_wer - ft_wer, 2), "CER (%)": round(base_cer - ft_cer, 2)},
    ])

    print("\n" + "=" * 50)
    print(table.to_string(index=False))
    print("=" * 50)

    table.to_csv("../results/comparison_table.csv", index=False)
    print("Saved to results/comparison_table.csv")


# ── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves():
    """
    Plot training loss and validation WER per epoch.
    """
    df = pd.DataFrame(TRAINING_LOGS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training loss
    ax1.plot(df["epoch"], df["train_loss"], marker="o", color="steelblue", linewidth=2)
    ax1.set_title("Training Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Validation WER
    ax2.plot(df["epoch"], df["val_wer"] * 100, marker="o", color="coral", linewidth=2)
    ax2.set_title("Validation WER per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("WER (%)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("../results/training_curves.png", dpi=150)
    print("Saved to results/training_curves.png")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}")

    # Load test data
    raw_test = loader(DATA_DIR, split="test")
    test_data = list(raw_test.select(range(min(TEST_SAMPLES, len(raw_test)))))

    # Load processor
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="azerbaijani", task="transcribe"
    )

    # ── Base model inference ──────────────────────────────────────────────────
    print("\nRunning base model inference...")
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)
    base_results = run_inference(base_model, processor, test_data)
    del base_model
    torch.cuda.empty_cache()

    # ── Fine-tuned model inference ────────────────────────────────────────────
    print("\nRunning fine-tuned model inference...")
    ft_model = WhisperForConditionalGeneration.from_pretrained(FINETUNED_MODEL).to(DEVICE)
    ft_results = run_inference(ft_model, processor, test_data)
    del ft_model
    torch.cuda.empty_cache()

    # ── Compare ───────────────────────────────────────────────────────────────
    compare_models(base_results, ft_results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_training_curves()

    print("\nDone! Check results/ folder.")


if __name__ == "__main__":
    main()