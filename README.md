# 🎙️ Azerbaijani ASR — Whisper Fine-Tuning

Automatic Speech Recognition (ASR) pipeline for the Azerbaijani language using OpenAI Whisper.

---

## 📌 Layihənin İzahatı

Bu layihə Mozilla Common Voice datasetindən istifadə edərək `openai/whisper-small` modelini Azərbaycan dili üçün fine-tune edir. Pipeline üç hissədən ibarətdir:

- **Hissə A** — Baza modelin inference-i və WER/CER qiymətləndirilməsi
- **Hissə B** — Whisper modelinin Azərbaycan dili üzrə fine-tuning-i və müqayisəsi
- **Hissə C** — Analitik hesabat (Azərbaycan dilində)

---

## 🤖 İstifadə Olunan Model və Parametrlər

| Parametr | Dəyər |
|---|---|
| Base model | `openai/whisper-small` |
| Dataset | Mozilla Common Voice 17.0 (az) |
| Train nümunələri | 150 |
| Validation nümunələri | 30 |
| Test nümunələri | 50 |
| Epoch sayı | 10 |
| Batch size | 8 |
| Learning rate | 1e-5 |
| Warmup steps | 50 |
| Optimizer | AdamW |
| Precision | fp16 (GPU) |
| Gradient checkpointing | Aktiv |

---

## 📊 WER/CER Nəticələri

| Model | WER (%) | CER (%) |
|---|---|---|
| Baza (whisper-small) | 65.32 | 18.49 |
| **Fine-tuned** | **51.63** | **12.92** |
| **Yaxşılaşma** | **↓ 13.69** | **↓ 5.57** |

> Best checkpoint: **Epoch 5** (Val WER: 52.34%)

---

## ⚙️ Setup

### 1. Conda mühitinin qurulması

```bash
conda create -n az-rec python=3.10 -y
conda activate az-rec
conda install -c conda-forge ffmpeg -y
```

### 2. Asılılıqların quraşdırılması

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets jiwer evaluate accelerate soundfile librosa pandas matplotlib
```

### 3. Dataset hazırlığı

Mozilla Common Voice Azerbaijani datasetini [buradan](https://commonvoice.mozilla.org/az/datasets) yükləyin və aşağıdakı strukturda yerləşdirin:

```
part_a/data/
├── train.tsv
├── dev.tsv
├── test.tsv
└── clips/
    ├── sample1.mp3
    ├── sample2.mp3
    └── ...
```

---

## 🚀 Kodu İşə Salmaq

### Hissə A — Inference və Qiymətləndirmə

```bash
# 1. Inference
cd part_a
python inference.py

# 2. WER/CER hesablama
python evaluate.py
```

### Hissə B — Fine-Tuning və Müqayisə

```bash
cd part_b

# 1. Fine-tuning
python fine_tuner.py

# 2. Baza vs Fine-tuned müqayisəsi
python compare.py
```

Nəticələr `results/` qovluğunda saxlanılır:
```
results/
├── inference_results.csv     # Part A inference nəticələri
├── evaluated_results.csv     # WER/CER per sample
├── comparison_table.csv      # Baza vs fine-tuned cədvəl
└── training_curves.png       # Training loss + val WER qrafiki
```

---

## 📁 Layihə Strukturu

```
az-stt-intern/
├── part_a/
│   ├── data_loader.py       # Dataset yükləmə modulu
│   ├── inference.py         # Baza model inference
│   └── evaluate.py          # WER/CER qiymətləndirməsi
├── part_b/
│   ├── dataset.py           # PyTorch Dataset + Data Collator
│   ├── fine_tuner.py        # Seq2SeqTrainer ilə fine-tuning
│   └── compare.py           # Baza vs fine-tuned müqayisəsi
├── results/
│   ├── comparison_table.csv
│   └── training_curves.png
├── report_c.pdf             # Analitik hesabat (Azərbaycan dilində)
├── requirements.txt
└── README.md
```

---

## 📈 Fine-Tuning Müqayisəsi

### Training Dinamikası

| Epoch | Train Loss | Val WER (%) |
|---|---|---|
| 1 | 2.620 | 60.75 |
| 2 | 1.356 | 57.01 |
| 3 | 0.688 | 54.67 |
| 4 | 0.381 | 53.27 |
| **5** | **0.240** | **52.34** ← best |
| 6 | 0.164 | 52.34 |
| 7 | 0.107 | 53.27 |
| 8 | 0.045 | 53.27 |
| 9 | 0.011 | 54.67 |
| 10 | 0.007 | 54.67 |

### Əsas Müşahidələr

- Fine-tuning **cəmi 150 nümunə** ilə WER-i **13.69%** azaltdı
- Overfitting **epoch 7-dən** başlayır — train loss azalır, val WER artır
- `load_best_model_at_end=True` ilə epoch 5-in modeli avtomatik saxlanıldı
- CER (Character Error Rate) WER-dən aşağıdır — model fonetik səviyyədə daha yaxşı öyrənir

---

## 🛠️ Requirements

```
torch
torchaudio
transformers>=4.40.0
datasets
jiwer
evaluate
accelerate
soundfile
librosa
pandas
matplotlib
```

---

## 📝 Qeydlər

- GPU tövsiyə olunur (used: RTX 4060 8GB)
- CPU ilə inference mümkündür, lakin fine-tuning çox yavaş olacaq
