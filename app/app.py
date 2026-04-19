"""
app/app.py

Joint Fault Transformer -- Bolt Loosening Severity Classification
Accepts a .mat file from the ORION-AE dataset, runs the Cross-Modal
Transformer to classify bolt loosening across 7 severity levels.

Usage:
    python app/app.py
"""

import sys
import traceback
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import torch
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from models.transformer.model import JointFaultTransformer
from utils.transforms import supervised_transform

# constants
REPO_ID       = 'mathewprasanth/JointFaultTransformerWeights'
WINDOW_SIZE   = 10_000
NUM_CLASSES   = 7
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    '05 cNm — Nearly free',
    '10 cNm — Very loose',
    '20 cNm — Significantly loose',
    '30 cNm — Half nominal torque',
    '40 cNm — Noticeable preload loss',
    '50 cNm — Slightly reduced preload',
    '60 cNm — Fully tight',
]

URGENCY = [
    '🔴 CRITICAL — Joint essentially unbolted',
    '🔴 URGENT — Almost no clamping force',
    '🟠 HIGH — Significant preload loss',
    '🟠 MODERATE — Joint starting to slip',
    '🟡 LOW — Noticeable preload loss',
    '🟢 MONITOR — Slightly reduced preload',
    '✅ NORMAL — Fully tight',
]

# download and load model once at startup
print('Downloading weights from Hugging Face...')
model_path   = hf_hub_download(repo_id=REPO_ID, filename='best_model.pt')
encoder_path = hf_hub_download(repo_id=REPO_ID, filename='simclr_encoder.pt')

print('Loading model...')
model = JointFaultTransformer(
    embedding_dim   = 128,
    num_heads       = 4,
    num_layers      = 2,
    num_classes     = NUM_CLASSES,
)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(DEVICE)
model.eval()
print('Model ready.')


def load_mat_file(filepath):
    mat     = scipy.io.loadmat(filepath)
    signals = np.stack([
        mat['A'].squeeze(),
        mat['B'].squeeze(),
        mat['C'].squeeze(),
    ], axis=0).astype(np.float32)
    return signals


def get_window(signals):
    n      = signals.shape[1]
    start  = max(0, n // 2 - WINDOW_SIZE // 2)
    window = signals[:, start:start + WINDOW_SIZE]
    return torch.from_numpy(window)


def plot_signals(signals):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    sensor_names = ['Sensor A (Micro80)', 'Sensor B (F50A)', 'Sensor C (Micro200HF)']
    colors       = ['#2196F3', '#4CAF50', '#FF9800']
    t = np.arange(50_000) / 5_000_000 * 1000

    for i, (ax, name, color) in enumerate(zip(axes, sensor_names, colors)):
        ax.plot(t, signals[i, :50_000], color=color, linewidth=0.5, alpha=0.8)
        ax.set_ylabel(name, fontsize=9)
        ax.set_ylim(-100, 100)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Raw AE Signals (first 10ms)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_attention(attn_weights):
    fig, ax = plt.subplots(figsize=(5, 4))
    weights = attn_weights[0].cpu().numpy()
    im = ax.imshow(weights, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Sensor A', 'Sensor B', 'Sensor C'])
    ax.set_yticklabels(['Sensor A', 'Sensor B', 'Sensor C'])
    ax.set_title('Cross-Sensor Attention Weights', fontweight='bold')
    plt.colorbar(im, ax=ax)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{weights[i, j]:.2f}',
                    ha='center', va='center', fontsize=10,
                    color='white' if weights[i, j] > 0.5 else 'black')
    plt.tight_layout()
    return fig


def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ['#f44336', '#ff5722', '#ff9800', '#ffc107', '#8bc34a', '#4caf50', '#2196f3']
    bars    = ax.barh(CLASS_NAMES, probs, color=colors, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Class Probabilities', fontweight='bold')
    for bar, prob in zip(bars, probs):
        ax.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f'{prob:.1%}', va='center', fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def predict(mat_file):
    if mat_file is None:
        return 'Please upload a .mat file from the ORION-AE dataset.', None, None, None

    try:
        signals = load_mat_file(mat_file.name)
        window  = get_window(signals)
        window  = supervised_transform(window.float())
        x       = window.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred   = int(probs.argmax())
            conf   = float(probs.max())
            attn   = model.get_attention_weights(x)

        result_text = (
            f'**Predicted Loosening Level:** {CLASS_NAMES[pred]}\n\n'
            f'**Confidence:** {conf:.1%}\n\n'
            f'**Status:** {URGENCY[pred]}'
        )

        fig_signals = plot_signals(signals)
        fig_probs   = plot_probabilities(probs)
        fig_attn    = plot_attention(attn)

        return result_text, fig_signals, fig_probs, fig_attn

    except Exception as e:
        traceback.print_exc()
        return f'Error: {str(e)}', None, None, None


with gr.Blocks(title='Joint Fault Transformer') as demo:

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown('# 🔩 Joint Fault Transformer')
            gr.Markdown(
                'Bolt loosening severity classification using 3 fused acoustic emission '
                'sensor streams. Upload a .mat file from the ORION-AE dataset to classify '
                'the bolt loosening level across 7 severity levels.'
            )
        with gr.Column(scale=1):
            gr.Markdown(
                "<div style='text-align: right;'>"
                "<strong>Mathew Prasanth</strong><br>AI/ML Engineer"
                "</div>"
            )

    gr.Markdown(
        '> **Pipeline:** 3 AE sensor streams → 3 independent 1D CNN encoders → '
        'Cross-Modal Transformer (cross-sensor attention) → 7-class bolt loosening classifier'
    )

    with gr.Row():
        with gr.Column(scale=1):
            mat_input   = gr.File(
                label      = 'Upload .mat file (ORION-AE dataset)',
                file_types = ['.mat'],
            )
            analyze_btn = gr.Button('Classify Loosening Level', variant='primary')

        with gr.Column(scale=2):
            result_text = gr.Markdown(label='Result')

    gr.Markdown('### Raw AE Signals')
    with gr.Row():
        signal_plot = gr.Plot(label='3 Sensor Streams')

    gr.Markdown('### Classification')
    with gr.Row():
        prob_plot = gr.Plot(label='Class Probabilities')

    gr.Markdown('### Cross-Sensor Attention — which sensors the model focused on')
    with gr.Row():
        attn_plot = gr.Plot(label='Attention Heatmap')

    analyze_btn.click(
        fn      = predict,
        inputs  = [mat_input],
        outputs = [result_text, signal_plot, prob_plot, attn_plot]
    )

    gr.Markdown('---')
    gr.Markdown(
        '**Model:** Cross-Modal Transformer with SimCLR contrastive pre-training | '
        '**Dataset:** ORION-AE benchmark (FEMTO-ST Institute) | '
        '**Accuracy:** 87% on held-out test windows across 5 experimental campaigns'
    )


demo.launch(
    server_name = '0.0.0.0',
    server_port = 7860,
)