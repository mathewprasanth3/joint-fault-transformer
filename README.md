# Joint Fault Transformer

Multi-sensor structural health monitoring system using deep learning to classify bolt loosening severity across 7 levels from acoustic emission signals — with cross-sensor attention explainability.

**Live Demo** → https://huggingface.co/spaces/mathewprasanth/JointFaultTransformer  
**Model Weights** → https://huggingface.co/mathewprasanth/JointFaultTransformerWeights

---

## What It Does

Bolt loosening in mechanical joints is a critical failure mode in structures and machinery. Traditional inspection requires manual torque checks or expensive monitoring hardware. This system classifies bolt loosening severity from raw acoustic emission signals recorded during vibration, enabling automated non-destructive monitoring.

The pipeline performs three tasks:
- Encodes raw acoustic emission signals from 3 sensors independently using 1D CNN encoders
- Fuses cross-sensor information using a Cross-Modal Transformer with multi-head self-attention
- Classifies bolt loosening severity into 7 levels from nearly free to fully tight

This reflects real-world structural health monitoring workflows where multi-sensor fusion and interpretability are critical for deployment confidence.

---

## Tech Stack

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

## Dataset

**ORION-AE** — Real experimental acoustic emission data from FEMTO-ST Institute, Besancon, France. Published on Harvard Dataverse (doi: 10.7910/DVN/FBRDU0).

| Property | Value |
|---|---|
| Campaigns | 5 (B, C, D, E, F) |
| Loosening levels | 7 (05 cNm to 60 cNm) |
| Files per level | ~10 files x 1 second each |
| Sampling rate | 5 MHz |
| Sensors | 3 AE sensors (Micro80, F50A, Micro200HF) |
| Total windows | ~338,000 (after windowing) |
| Dataset size | ~9 GB |

---

## Pipeline Architecture

```
Raw AE Signal (3 sensors x 10,000 samples = 2ms window)
-> TripleEncoder (3 x 1D CNN, independent weights per sensor)
-> Cross-Modal Transformer (cross-sensor attention, 4 heads, 2 layers)
-> Classification Head (Linear -> ReLU -> Dropout -> Linear)
-> Output: 7-class bolt loosening severity prediction
```

---

## Model Architecture

### Phase 1 — SimCLR Contrastive Pre-training
- Training method: Self-supervised, no labels
- Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- Augmentations: Noise, random scale, random crop
- Output: Pre-trained encoder weights

### Phase 2 — Supervised Fine-tuning
- Loss: CrossEntropyLoss
- Optimiser: Adam lr=1e-4
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=15
- Test accuracy: **87% on held-out test windows**

### Model Parameters

| Component | Parameters |
|---|---|
| TripleEncoder (3 x CNNEncoder) | 156,672 |
| CrossModalTransformer | 397,184 |
| Classification Head | ~8,711 |
| **Total** | **562,567** |

---

## Key Training Decisions

- Independent encoder weights per sensor — each sensor has different frequency range (50kHz-1MHz)
- SimCLR pre-training before supervised fine-tuning — only 340 labelled files, labels not needed for Phase 1
- AdaptiveAvgPool1d(1) — collapses variable time dimension robustly regardless of file length
- Campaign-wise caching — pre-processes all 338,524 windows to disk, 10-100x training speedup
- Cross-campaign domain shift identified as deployment challenge — calibration fine-tuning proposed as solution

---

## Project Structure

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

## Inference Pipeline

1. Upload .mat file from ORION-AE dataset
2. Extract middle 2ms window from all 3 AE sensor channels
3. Normalise each channel to [-1, 1]
4. Pass through TripleEncoder — 3 independent 1D CNN encoders
5. Fuse sensor embeddings through Cross-Modal Transformer
6. Classify into 7 bolt loosening severity levels
7. Output includes prediction, confidence, class probabilities, and cross-sensor attention heatmap

---

## Deployment

- Hosted on Hugging Face Spaces
- Model weights stored on Hugging Face Model Hub
- Downloaded dynamically at runtime
- Runs on CPU (HF Spaces constraint)

---

## Key Engineering Decisions

- Three independent 1D CNN encoders rather than one shared encoder — preserves sensor-specific frequency characteristics
- Projection head used only during SimCLR Phase 1 then discarded — encoder output used directly for Phase 2
- Positional encoding on sensor tokens — tells transformer which token is sensor A, B, or C
- supervised_transform applied consistently in both training and evaluation — prevents normalisation mismatch

---

## Run Locally

```bash
git clone https://github.com/mathewprasanth3/joint-fault-transformer.git
cd joint-fault-transformer
pip install -r requirements.txt
python app/app.py
```

---

## Training

```bash
# Phase 1 -- SimCLR contrastive pre-training (no labels)
python models/encoder/train.py

# Phase 2 -- supervised fine-tuning
python models/transformer/train.py
```

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | 87% |
| Val accuracy (best epoch) | 93% |
| F1 score (macro) | 0.87 |
| Total parameters | 562,567 |
| Training windows | 268,627 |

---

## Author

**Mathew Prasanth, PE**  
AI/ML Engineer  
[https://www.linkedin.com/in/mathewprasanth/](https://www.linkedin.com/in/mathewprasanth/)  
[https://huggingface.co/spaces/mathewprasanth/JointFaultTransformer](https://huggingface.co/spaces/mathewprasanth/JointFaultTransformer)

AWS Certified Cloud Practitioner · AWS Certified Machine Learning Specialty
