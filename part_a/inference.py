import torch
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio, Features, Value
import pandas as pd
from data_loader import loader

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Config ──────────────────────────────────────────
DATA_DIR = "data"
MODEL_NAME = "openai/whisper-small"
SAMPLE_RATE = 16000
NUM_SAMPLES = 100  # run on first 100 test samples
# ────────────────────────────────────────────────────

# Load model and processor
print("Loading model...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Force Azerbaijani language
forced_decoder_ids = processor.get_decoder_prompt_ids(language="azerbaijani", task="transcribe")

ds = loader(data_dir=DATA_DIR, split="test")
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))

# ── Inference ────────────────────────────────────────
results = []

for i, sample in enumerate(ds):
    audio_path = sample["path"]
    reference  = sample["sentence"]

    try:
        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Prepare input
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs["input_features"].to(device)  


        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )

        hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        results.append({
            "reference": reference,
            "hypothesis": hypothesis,
            "audio_path": audio_path
        })

        if i % 10 == 0:
            print(f"[{i+1}/{len(ds)}] Done")

    except Exception as e:
        print(f"Skipped {audio_path}: {e}")

# Save results for evaluation
df = pd.DataFrame(results)
df.to_csv("results/inference_results.csv", index=False)
print(f"\nSaved {len(df)} results to results/inference_results.csv")