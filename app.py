#!/usr/bin/env python3
"""
WaterPrint - Water Mass Classification using Isotopic Fingerprints
Simple Gradio app for HuggingFace Spaces
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# Load model
MODEL_PATH = Path(__file__).parent / "models" / "waterprint_model.joblib"

COLORS = {'NADW': '#e41a1c', 'AABW': '#377eb8', 'AAIW': '#4daf4a', 'CDW': '#984ea3'}
DESCRIPTIONS = {
    'NADW': 'North Atlantic Deep Water - Recently ventilated, young',
    'AABW': 'Antarctic Bottom Water - Oldest, longest isolation',
    'AAIW': 'Antarctic Intermediate Water - Fresh, recently ventilated',
    'CDW': 'Circumpolar Deep Water - Southern Ocean, aged',
}

# Global model
model_data = None

def load_model():
    global model_data
    if model_data is None and MODEL_PATH.exists():
        model_data = joblib.load(MODEL_PATH)
    return model_data is not None

def classify(d14c, d13c):
    """Classify a single sample."""
    if d14c is None or d13c is None:
        return "Please enter both values", None, None

    if not load_model():
        return "Model not loaded", None, None

    # Predict
    X = np.array([[d14c, d13c]])
    X_scaled = model_data['scaler_isotope'].transform(X)
    proba = model_data['model_isotope'].predict_proba(X_scaled)[0]
    classes = model_data['label_encoder'].classes_

    pred_idx = np.argmax(proba)
    pred_class = classes[pred_idx]
    confidence = proba[pred_idx]

    # Result text
    result = f"## Prediction: {pred_class}\n\n"
    result += f"**Confidence:** {confidence:.1%}\n\n"
    result += f"*{DESCRIPTIONS.get(pred_class, '')}*\n\n"
    result += "### Probabilities\n"
    for i, c in enumerate(classes):
        result += f"- {c}: {proba[i]:.1%}\n"

    # Bar chart
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    colors = [COLORS.get(c, 'gray') for c in classes]
    ax1.barh(classes, proba, color=colors)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Probability')
    ax1.set_title('Classification Result')
    plt.tight_layout()

    # Scatter plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    training = model_data['training_data']
    for wm in COLORS:
        mask = training['water_mass'] == wm
        if mask.sum() > 0:
            ax2.scatter(training.loc[mask, 'delta14c'], training.loc[mask, 'delta13c'],
                       c=COLORS[wm], label=wm, alpha=0.3, s=8)
    ax2.scatter([d14c], [d13c], c='black', s=200, marker='*',
               edgecolors='white', linewidths=2, zorder=10, label='Your sample')
    ax2.set_xlabel('Δ¹⁴C (‰)')
    ax2.set_ylabel('δ¹³C (‰)')
    ax2.set_title('Sample Position in Isotope Space')
    ax2.legend()
    plt.tight_layout()

    return result, fig1, fig2

def show_data():
    """Show training data visualization."""
    if not load_model():
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    training = model_data['training_data']
    for wm in COLORS:
        mask = training['water_mass'] == wm
        if mask.sum() > 0:
            ax.scatter(training.loc[mask, 'delta14c'], training.loc[mask, 'delta13c'],
                      c=COLORS[wm], label=f"{wm} (n={mask.sum()})", alpha=0.5, s=15)
    ax.set_xlabel('Δ¹⁴C (‰)')
    ax.set_ylabel('δ¹³C (‰)')
    ax.set_title('GLODAP Training Data - Isotope Fingerprints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Build interface
with gr.Blocks(title="WaterPrint", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# WaterPrint: Water Mass Classification")
    gr.Markdown("Classify ocean water masses using isotopic fingerprints (Δ¹⁴C, δ¹³C)")

    with gr.Tab("Classify"):
        with gr.Row():
            with gr.Column(scale=1):
                d14c = gr.Number(label="Δ¹⁴C (‰)", value=-120, info="Range: -200 to 0")
                d13c = gr.Number(label="δ¹³C (‰)", value=0.5, info="Range: -1 to 2")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column(scale=2):
                output = gr.Markdown()
        with gr.Row():
            plot1 = gr.Plot(label="Probabilities")
            plot2 = gr.Plot(label="Position")
        btn.click(classify, [d14c, d13c], [output, plot1, plot2])

    with gr.Tab("Data"):
        gr.Markdown("### Training Data Distribution")
        data_plot = gr.Plot()
        gr.Button("Show Data").click(show_data, [], [data_plot])

    gr.Markdown("---")
    gr.Markdown("**WaterPrint** | Gruber (2025) | GLODAP v2.2023 | Accuracy: 74%")

if __name__ == "__main__":
    demo.launch()
