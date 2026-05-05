# ⚙️ Joint Fault Transformer

Multi-sensor structural health monitoring system using deep learning to classify bolt loosening severity across 7 levels from acoustic emission signals — with SimCLR pre-training and cross-sensor attention explainability.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/mathewprasanth/JointFaultTransformer)
[![Model Weights](https://img.shields.io/badge/Weights-HuggingFace-blue)](https://huggingface.co/mathewprasanth/JointFaultTransformerWeights)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)](https://pytorch.org)

---

## 📊 Results

| Metric | Value |
|---|---|
| Within-campaign accuracy | 90% |
| Val accuracy (best epoch) | 93% |
| F1 score (macro) | 0.87 |
| Cross-campaign accuracy (held-out F) | 74% |
| Post-calibration fine-tune (30% target data) | 67% → recovered |
| Total parameters | 562,567 |
| Training windows | 268,627 |
| Loosening levels classified | 7 |

---

## 🧠 What It Does

Bolt loosening in mechanical joints is a critical failure mode in structures and machinery. Traditional inspection requires manual torque checks or expensive monitoring hardware. This system classifies bolt loosening severity from raw acoustic emission signals recorded during vibration, enabling automated non-destructive monitoring.

The pipeline performs three tasks:
- Encodes raw acoustic emission signals from 3 sensors independently using 1D CNN encoders
- Fuses cross-sensor information using a Cross-Modal Transformer with multi-head self-attention
- Classifies bolt loosening severity into 7 levels from nearly free to fully tight

This reflects real-world structural health monitoring workflows where multi-sensor fusion and interpretability are critical for deployment confidence.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Signal Encoder | 1D CNN (3 independent encoders, one per AE sensor) |
| Fusion Model | Cross-Modal Transformer (4-head self-attention, 2 layers) |
| Pre-training | SimCLR contrastive learning (Phase 1, no labels) |
| Backend | PyTorch |
| UI | Gradio |
| Hosting | Hugging Face Spaces |
| Model Storage | Hugging Face Model Hub |
| Signal Processing | SciPy, NumPy |

---

## 🏗️ Pipeline Architecture

```
Raw AE Signal (3 sensors × 10,000 samples = 2ms window)
→ TripleEncoder (3 × 1D CNN, independent weights per sensor)
→ Cross-Modal Transformer (cross-sensor attention, 4 heads, 2 layers)
→ Classification Head (Linear → ReLU → Dropout → Linear)
→ Output: 7-class bolt loosening severity prediction
```

---

## 🔬 Two-Phase Training

### Phase 1 — SimCLR Contrastive Pre-training (no labels)
- Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- Augmentations: noise injection, random scale, random crop
- Output: pre-trained encoder weights capturing signal structure

### Phase 2 — Supervised Fine-tuning
- Loss: CrossEntropyLoss
- Optimiser: Adam lr=1e-4
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=15

### Model Parameters

| Component | Parameters |
|---|---|
| TripleEncoder (3 × CNNEncoder) | 156,672 |
| CrossModalTransformer | 397,184 |
| Classification Head | 8,711 |
| **Total** | **562,567** |

---

## 🔑 Key Engineering Decisions

- Independent encoder weights per sensor — each sensor has a different frequency range (50kHz–1MHz), shared weights would lose sensor-specific information
- SimCLR pre-training before supervised fine-tuning — only 340 labelled files available, Phase 1 requires no labels
- AdaptiveAvgPool1d(1) — collapses variable time dimension robustly regardless of file length
- Campaign-wise caching — pre-processes all 338,524 windows to disk for 10-100x training speedup
- Cross-campaign domain shift empirically confirmed (16% drop on held-out campaign F vs 90% within-campaign) — calibration fine-tuning on 30% of target data recovers to 67%

---

## 📁 Project Structure

```
joint-fault-transformer/
├── app/
│   └── app.py
├── requirements.txt
├── models/
│   ├── encoder/
│   │   ├── model.py
│   │   └── train.py
│   └── transformer/
│       ├── model.py
│       └── train.py
├── utils/
│   ├── dataset.py
│   └── transforms.py
├── notebooks/
│   └── 01_data_exploration.ipynb
└── data/
    ├── raw/
    └── processed/
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/mathewprasanth/joint-fault-transformer.git
cd joint-fault-transformer
pip install -r requirements.txt
python app/app.py
```

## Training

```bash
# Phase 1 — SimCLR contrastive pre-training (no labels)
python models/encoder/train.py

# Phase 2 — supervised fine-tuning
python models/transformer/train.py
```

---

## ⚠️ Limitations

- Cross-campaign domain shift reduces accuracy from 90% to 74% on unseen campaigns — real deployment requires calibration fine-tuning on a small amount of target campaign data
- ORION-AE dataset is lab-controlled — real structural monitoring environments introduce additional noise sources
- 7-level classification requires careful torque calibration during data collection

---

## 👤 Author

**Mathew Prasanth, P.E.**
AI/ML Engineer | U.S. Licensed Professional Engineer
[LinkedIn](https://www.linkedin.com/in/mathewprasanth/) · [Live Demo](https://huggingface.co/spaces/mathewprasanth/JointFaultTransformer)

*AWS Certified ML Specialty · AWS Cloud Practitioner*
