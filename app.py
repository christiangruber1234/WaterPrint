#!/usr/bin/env python3
"""
WaterPrint - Water Mass Classification using Isotopic Fingerprints
Gradio 6.x app for HuggingFace Spaces
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.neighbors import NearestNeighbors
import folium

# Load model
MODEL_PATH = Path(__file__).parent / "models" / "waterprint_model.joblib"

COLORS = {'NADW': '#e41a1c', 'AABW': '#377eb8', 'AAIW': '#4daf4a', 'CDW': '#984ea3'}
DESCRIPTIONS = {
    'NADW': 'North Atlantic Deep Water - Recently ventilated, relatively young (mean Δ¹⁴C: −86‰)',
    'AABW': 'Antarctic Bottom Water - Oldest water mass, longest isolation (mean Δ¹⁴C: −177‰)',
    'AAIW': 'Antarctic Intermediate Water - Fresh, recently ventilated (mean Δ¹⁴C: −70‰)',
    'CDW': 'Circumpolar Deep Water - Southern Ocean, aged water mass (mean Δ¹⁴C: −157‰)',
}

# Global model and nearest neighbors
model_data = None
nn_model = None

def load_model():
    global model_data, nn_model
    if model_data is None and MODEL_PATH.exists():
        model_data = joblib.load(MODEL_PATH)
        X = model_data['training_data'][['delta14c', 'delta13c']].values
        X_scaled = model_data['scaler_isotope'].transform(X)
        nn_model = NearestNeighbors(n_neighbors=20, metric='euclidean')
        nn_model.fit(X_scaled)
    return model_data is not None

# ============ TAB 1: CLASSIFY ============
def classify(d14c, d13c):
    if d14c is None or d13c is None:
        return "Please enter both values", None, None
    if not load_model():
        return "Model not loaded", None, None

    X = np.array([[d14c, d13c]])
    X_scaled = model_data['scaler_isotope'].transform(X)
    proba = model_data['model_isotope'].predict_proba(X_scaled)[0]
    classes = model_data['label_encoder'].classes_

    pred_idx = np.argmax(proba)
    pred_class = classes[pred_idx]
    confidence = proba[pred_idx]

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

# ============ TAB 2: BATCH ============
def classify_batch(file):
    if file is None:
        return "Please upload a CSV file", None
    if not load_model():
        return "Model not loaded", None

    try:
        df = pd.read_csv(file.name)

        for col in df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'delta14' in col_lower or 'd14c' in col_lower:
                df = df.rename(columns={col: 'delta14c'})
            elif 'delta13' in col_lower or 'd13c' in col_lower:
                df = df.rename(columns={col: 'delta13c'})

        if 'delta14c' not in df.columns or 'delta13c' not in df.columns:
            return "Could not find Δ¹⁴C and δ¹³C columns", None

        results = []
        classes = model_data['label_encoder'].classes_

        for idx, row in df.iterrows():
            d14c = row.get('delta14c')
            d13c = row.get('delta13c')

            if pd.isna(d14c) or pd.isna(d13c):
                results.append({'sample': idx, 'd14c': None, 'd13c': None, 'prediction': 'N/A', 'confidence': 0})
                continue

            X = np.array([[d14c, d13c]])
            X_scaled = model_data['scaler_isotope'].transform(X)
            proba = model_data['model_isotope'].predict_proba(X_scaled)[0]
            pred_idx = np.argmax(proba)

            results.append({
                'sample': idx, 'd14c': d14c, 'd13c': d13c,
                'prediction': classes[pred_idx], 'confidence': proba[pred_idx]
            })

        results_df = pd.DataFrame(results)
        valid = results_df[results_df['prediction'] != 'N/A']

        summary = f"**Total samples:** {len(results)}\n\n"
        summary += f"**Classified:** {len(valid)}\n\n"
        summary += "**Distribution:**\n"
        for wm, count in valid['prediction'].value_counts().items():
            summary += f"- {wm}: {count}\n"

        summary += "\n### Results\n\n"
        summary += "| # | Δ¹⁴C | δ¹³C | Prediction | Confidence |\n"
        summary += "|---|------|------|------------|------------|\n"
        for _, row in results_df.head(20).iterrows():
            if row['prediction'] != 'N/A':
                summary += f"| {row['sample']} | {row['d14c']:.1f} | {row['d13c']:.2f} | {row['prediction']} | {row['confidence']:.1%} |\n"
        if len(results_df) > 20:
            summary += f"\n*Showing first 20 of {len(results_df)} samples*\n"

        fig, ax = plt.subplots(figsize=(10, 6))
        for wm in COLORS:
            mask = valid['prediction'] == wm
            if mask.sum() > 0:
                ax.scatter(valid.loc[mask, 'd14c'], valid.loc[mask, 'd13c'],
                          c=COLORS[wm], label=wm, alpha=0.7, s=50)
        ax.set_xlabel('Δ¹⁴C (‰)')
        ax.set_ylabel('δ¹³C (‰)')
        ax.set_title('Classified Samples')
        ax.legend()
        plt.tight_layout()

        return summary, fig
    except Exception as e:
        return f"Error: {str(e)}", None

# ============ TAB 3: VISUALIZATION ============
def show_isotope_space(highlight):
    if not load_model():
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    training = model_data['training_data']
    for wm in COLORS:
        mask = training['water_mass'] == wm
        if mask.sum() > 0:
            alpha = 0.7 if wm == highlight or highlight == "All" else 0.15
            size = 25 if wm == highlight or highlight == "All" else 8
            ax.scatter(training.loc[mask, 'delta14c'], training.loc[mask, 'delta13c'],
                      c=COLORS[wm], label=f"{wm} (n={mask.sum()})", alpha=alpha, s=size)
    ax.set_xlabel('Δ¹⁴C (‰)')
    ax.set_ylabel('δ¹³C (‰)')
    ax.set_title('Isotope Fingerprint Space - GLODAP Training Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def show_ts_diagram(highlight):
    if not load_model():
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    training = model_data['training_data']
    for wm in COLORS:
        mask = training['water_mass'] == wm
        if mask.sum() > 0:
            subset = training.loc[mask].dropna(subset=['theta', 'salinity'])
            if len(subset) > 0:
                alpha = 0.7 if wm == highlight or highlight == "All" else 0.15
                size = 25 if wm == highlight or highlight == "All" else 8
                ax.scatter(subset['salinity'], subset['theta'],
                          c=COLORS[wm], label=f"{wm} (n={len(subset)})", alpha=alpha, s=size)
    ax.set_xlabel('Salinity (PSU)')
    ax.set_ylabel('Potential Temperature θ (°C)')
    ax.set_title('T-S Diagram - GLODAP Training Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ============ TAB 4: EXPLAINABILITY ============
def explain(d14c, d13c):
    if d14c is None or d13c is None:
        return "Please enter both values", None
    if not load_model():
        return "Model not loaded", None

    X = np.array([[d14c, d13c]])
    X_scaled = model_data['scaler_isotope'].transform(X)
    proba = model_data['model_isotope'].predict_proba(X_scaled)[0]
    classes = model_data['label_encoder'].classes_
    pred_class = classes[np.argmax(proba)]

    training = model_data['training_data']
    wm_stats = training.groupby('water_mass')[['delta14c', 'delta13c']].mean()

    result = f"**Input:** Δ¹⁴C = {d14c}‰, δ¹³C = {d13c}‰\n\n"
    result += f"**Predicted:** {pred_class} ({proba.max():.1%})\n\n"
    result += "### Distance to Water Mass Centers\n\n"

    distances = []
    for wm in ['NADW', 'AABW', 'AAIW', 'CDW']:
        if wm in wm_stats.index:
            mean_d14c = wm_stats.loc[wm, 'delta14c']
            mean_d13c = wm_stats.loc[wm, 'delta13c']
            dist = np.sqrt((d14c - mean_d14c)**2 + (d13c - mean_d13c)**2)
            distances.append((wm, dist, mean_d14c))
            marker = " ← **Best Match**" if wm == pred_class else ""
            result += f"- {wm}: {dist:.1f} (center Δ¹⁴C: {mean_d14c:.0f}‰){marker}\n"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for wm in COLORS:
        mask = training['water_mass'] == wm
        if mask.sum() > 0:
            axes[0].scatter(training.loc[mask, 'delta14c'], training.loc[mask, 'delta13c'],
                           c=COLORS[wm], label=wm, alpha=0.3, s=10)
    axes[0].scatter([d14c], [d13c], c='black', s=200, marker='*', edgecolors='white', linewidths=2, zorder=10)
    axes[0].set_xlabel('Δ¹⁴C (‰)')
    axes[0].set_ylabel('δ¹³C (‰)')
    axes[0].set_title('Sample Position')
    axes[0].legend()

    wm_names = [d[0] for d in distances]
    dists = [d[1] for d in distances]
    colors = [COLORS[wm] for wm in wm_names]
    axes[1].barh(wm_names, dists, color=colors)
    axes[1].set_xlabel('Distance to Centroid')
    axes[1].set_title('Distance to Each Water Mass')
    plt.tight_layout()
    return result, fig

# ============ TAB 5: SIMILAR SAMPLES ============
def create_sample_map(similar):
    """Create an interactive Folium map showing similar sample locations."""
    # Check for lat/lon columns
    if 'lat' not in similar.columns or 'lon' not in similar.columns:
        return None
    valid_coords = similar.dropna(subset=['lat', 'lon'])
    if len(valid_coords) == 0:
        return None

    center_lat = valid_coords['lat'].mean()
    center_lon = valid_coords['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='CartoDB positron',
                   width='100%', height=400)

    wm_colors = {'NADW': 'red', 'AABW': 'blue', 'AAIW': 'green', 'CDW': 'purple'}
    max_dist = similar['distance'].max() if 'distance' in similar.columns else 1

    for i, (_, row) in enumerate(valid_coords.iterrows(), 1):
        wm = row['water_mass']
        color = wm_colors.get(wm, 'gray')
        # Size inversely proportional to distance (closer = bigger)
        dist = row.get('distance', 0.5)
        similarity = 1 - (dist / max_dist) if max_dist > 0 else 0.5
        radius = 6 + (similarity * 8)

        popup = f"#{i} {wm}<br>Δ¹⁴C: {row['delta14c']:.1f}‰<br>δ¹³C: {row['delta13c']:.2f}‰"
        if 'depth' in row and pd.notna(row['depth']):
            popup += f"<br>Depth: {row['depth']:.0f}m"

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            popup=popup,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)

    return m


def find_similar(d14c, d13c, n_neighbors):
    if d14c is None or d13c is None:
        return "Please enter both values", None, "<p>Please enter values</p>"
    if not load_model():
        return "Model not loaded", None, "<p>Model not loaded</p>"

    n = int(n_neighbors)
    X = np.array([[d14c, d13c]])
    X_scaled = model_data['scaler_isotope'].transform(X)
    distances, indices = nn_model.kneighbors(X_scaled, n_neighbors=n)

    training = model_data['training_data']
    similar = training.iloc[indices[0]].copy()
    similar['distance'] = distances[0]

    result = f"**Your input:** Δ¹⁴C = {d14c}‰, δ¹³C = {d13c}‰\n\n"
    result += f"### Top {n} Similar Samples\n\n"
    for i, (_, row) in enumerate(similar.iterrows(), 1):
        result += f"{i}. **{row['water_mass']}** - Δ¹⁴C: {row['delta14c']:.1f}‰, δ¹³C: {row['delta13c']:.2f}‰"
        if 'depth' in row and pd.notna(row['depth']):
            result += f", Depth: {row['depth']:.0f}m"
        if 'lat' in row and 'lon' in row and pd.notna(row['lat']):
            result += f" ({row['lat']:.1f}°, {row['lon']:.1f}°)"
        result += "\n"

    result += "\n### Distribution\n"
    for wm, count in similar['water_mass'].value_counts().items():
        result += f"- {wm}: {count}\n"

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for wm in COLORS:
        mask = training['water_mass'] == wm
        if mask.sum() > 0:
            ax.scatter(training.loc[mask, 'delta14c'], training.loc[mask, 'delta13c'], c=COLORS[wm], alpha=0.1, s=5)
    for wm in COLORS:
        mask = similar['water_mass'] == wm
        if mask.sum() > 0:
            ax.scatter(similar.loc[mask, 'delta14c'], similar.loc[mask, 'delta13c'],
                      c=COLORS[wm], label=wm, alpha=0.9, s=100, edgecolors='black', linewidths=1)
    ax.scatter([d14c], [d13c], c='yellow', s=300, marker='*', edgecolors='black', linewidths=2, zorder=10, label='Your sample')
    ax.set_xlabel('Δ¹⁴C (‰)')
    ax.set_ylabel('δ¹³C (‰)')
    ax.set_title(f'Your Sample and {n} Most Similar Samples')
    ax.legend()
    plt.tight_layout()

    # Create map
    sample_map = create_sample_map(similar)
    if sample_map:
        map_html = sample_map._repr_html_()
    else:
        map_html = "<p style='padding:20px; color:#666;'>No geographic coordinates available for these samples.</p>"

    return result, fig, map_html

# ============ TAB 6: UNCERTAINTY ============
def uncertainty(d14c, d13c):
    if d14c is None or d13c is None:
        return "Please enter both values", None
    if not load_model():
        return "Model not loaded", None

    X = np.array([[d14c, d13c]])
    X_scaled = model_data['scaler_isotope'].transform(X)
    proba = model_data['model_isotope'].predict_proba(X_scaled)[0]
    classes = model_data['label_encoder'].classes_
    pred_class = classes[np.argmax(proba)]

    entropy = -np.sum(proba * np.log(proba + 1e-10))
    norm_entropy = entropy / np.log(len(proba))
    sorted_proba = np.sort(proba)[::-1]
    margin = sorted_proba[0] - sorted_proba[1]
    distances, _ = nn_model.kneighbors(X_scaled, n_neighbors=5)
    mean_dist = np.mean(distances)

    if norm_entropy < 0.3:
        status = "LOW - Model is confident"
    elif norm_entropy < 0.6:
        status = "MODERATE - Some ambiguity"
    else:
        status = "HIGH - Model is uncertain"

    result = f"**Input:** Δ¹⁴C = {d14c}‰, δ¹³C = {d13c}‰\n\n"
    result += f"**Predicted:** {pred_class} ({proba.max():.1%})\n\n"
    result += f"### Uncertainty: {status}\n\n"
    result += "**Metrics:**\n"
    result += f"- Normalized Entropy: {norm_entropy:.2f}\n"
    result += f"- Confidence Margin: {margin:.2f}\n"
    result += f"- Distance to Neighbors: {mean_dist:.2f}\n\n"
    if margin < 0.2:
        result += "⚠️ Sample is near classification boundary\n"
    if mean_dist > 2.0:
        result += "⚠️ Sample is far from training data\n"
    result += "\n### Probabilities\n"
    for i, c in enumerate(classes):
        result += f"- {c}: {proba[i]:.1%}\n"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [COLORS.get(c, 'gray') for c in classes]
    axes[0].bar(classes, proba, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='50%')
    axes[0].axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Random')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Classification Probabilities')
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    training = model_data['training_data']
    for wm in COLORS:
        mask = training['water_mass'] == wm
        if mask.sum() > 0:
            axes[1].scatter(training.loc[mask, 'delta14c'], training.loc[mask, 'delta13c'],
                           c=COLORS[wm], alpha=0.3, s=10, label=wm)
    axes[1].scatter([d14c], [d13c], c='black', s=200, marker='*', edgecolors='white', linewidths=2, zorder=10)
    axes[1].set_xlabel('Δ¹⁴C (‰)')
    axes[1].set_ylabel('δ¹³C (‰)')
    axes[1].set_title('Sample Position')
    axes[1].legend()
    plt.tight_layout()
    return result, fig

# ============ BUILD INTERFACE (Gradio 6.x syntax) ============
with gr.Blocks(title="WaterPrint") as demo:
    gr.Markdown("# WaterPrint: Water Mass Classification")
    gr.Markdown("Classify ocean water masses using isotopic fingerprints (Δ¹⁴C, δ¹³C) — no geographic coordinates required")

    with gr.Tabs():
        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    d14c_1 = gr.Number(label="Δ¹⁴C (‰)", value=-120)
                    d13c_1 = gr.Number(label="δ¹³C (‰)", value=0.5)
                    btn1 = gr.Button("Classify", variant="primary")
                with gr.Column(scale=2):
                    out1 = gr.Markdown()
            with gr.Row():
                plot1a = gr.Plot(label="Probabilities")
                plot1b = gr.Plot(label="Position")
            btn1.click(classify, [d14c_1, d13c_1], [out1, plot1a, plot1b])

        with gr.Tab("Batch"):
            gr.Markdown("Upload CSV with `delta14c` and `delta13c` columns")
            file_in = gr.File(label="Upload CSV", file_types=[".csv"])
            btn2 = gr.Button("Process", variant="primary")
            out2 = gr.Markdown()
            plot2 = gr.Plot()
            btn2.click(classify_batch, [file_in], [out2, plot2])

        with gr.Tab("Visualization"):
            highlight = gr.Dropdown(["All", "NADW", "AABW", "AAIW", "CDW"], value="All", label="Highlight")
            with gr.Row():
                plot3a = gr.Plot(label="Isotope Space")
                plot3b = gr.Plot(label="T-S Diagram")
            with gr.Row():
                btn3a = gr.Button("Show Isotope Space")
                btn3b = gr.Button("Show T-S Diagram")
            btn3a.click(show_isotope_space, [highlight], [plot3a])
            btn3b.click(show_ts_diagram, [highlight], [plot3b])

        with gr.Tab("Explainability"):
            with gr.Row():
                with gr.Column(scale=1):
                    d14c_4 = gr.Number(label="Δ¹⁴C (‰)", value=-120)
                    d13c_4 = gr.Number(label="δ¹³C (‰)", value=0.5)
                    btn4 = gr.Button("Explain", variant="primary")
                with gr.Column(scale=2):
                    out4 = gr.Markdown()
            plot4 = gr.Plot()
            btn4.click(explain, [d14c_4, d13c_4], [out4, plot4])

        with gr.Tab("Similar Samples"):
            with gr.Row():
                with gr.Column(scale=1):
                    d14c_5 = gr.Number(label="Δ¹⁴C (‰)", value=-120)
                    d13c_5 = gr.Number(label="δ¹³C (‰)", value=0.5)
                    n_sim = gr.Slider(3, 20, value=5, step=1, label="Number of samples")
                    btn5 = gr.Button("Find Similar", variant="primary")
                with gr.Column(scale=2):
                    out5 = gr.Markdown()
            with gr.Row():
                plot5 = gr.Plot(label="Isotope Space")
                map5 = gr.HTML(label="Sample Locations")
            btn5.click(find_similar, [d14c_5, d13c_5, n_sim], [out5, plot5, map5])

        with gr.Tab("Uncertainty"):
            with gr.Row():
                with gr.Column(scale=1):
                    d14c_6 = gr.Number(label="Δ¹⁴C (‰)", value=-120)
                    d13c_6 = gr.Number(label="δ¹³C (‰)", value=0.5)
                    btn6 = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=2):
                    out6 = gr.Markdown()
            plot6 = gr.Plot()
            btn6.click(uncertainty, [d14c_6, d13c_6], [out6, plot6])

    gr.Markdown("---")
    gr.Markdown("**WaterPrint** | Gruber (2025) | Data: GLODAP v2.2023 | Isotope-only accuracy: 74.1% ± 1.7%")

if __name__ == "__main__":
    demo.launch()
