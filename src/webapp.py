#!/usr/bin/env python3
"""
WaterPrint Web Application
==========================

Interactive Gradio-based web interface for water mass classification.
Clean scientific design with light theme.

Author: ForamAI Project
Date: 2025-12-28
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report
import xgboost as xgb
import folium

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Water mass colors
WATER_MASS_COLORS = {
    'NADW': '#e41a1c',
    'AABW': '#377eb8',
    'AAIW': '#4daf4a',
    'CDW': '#984ea3',
}

WATER_MASS_DESCRIPTIONS = {
    'NADW': 'North Atlantic Deep Water - Recently ventilated, relatively young (mean Δ¹⁴C: −86‰)',
    'AABW': 'Antarctic Bottom Water - Oldest water mass, longest isolation (mean Δ¹⁴C: −177‰)',
    'AAIW': 'Antarctic Intermediate Water - Fresh, recently ventilated (mean Δ¹⁴C: −70‰)',
    'CDW': 'Circumpolar Deep Water - Southern Ocean, aged water mass (mean Δ¹⁴C: −157‰)',
}


class WaterPrintModel:
    """Wrapper class for the WaterPrint classification model."""

    def __init__(self):
        self.model_full = None
        self.model_isotope = None
        self.scaler_full = None
        self.scaler_isotope = None
        self.label_encoder = None
        self.training_data = None
        self.feature_cols_full = None
        self.feature_cols_isotope = ['delta14c', 'delta13c']
        self.nn_model = None

    def load_or_train(self):
        """Load existing model or train new one."""
        model_path = MODELS_DIR / "waterprint_model.joblib"

        if model_path.exists():
            print("Loading existing model...")
            saved = joblib.load(model_path)
            self.model_full = saved['model_full']
            self.model_isotope = saved['model_isotope']
            self.scaler_full = saved['scaler_full']
            self.scaler_isotope = saved['scaler_isotope']
            self.label_encoder = saved['label_encoder']
            self.training_data = saved['training_data']
            self.feature_cols_full = saved['feature_cols_full']
            self._fit_nn_model()
            return True
        else:
            print("Training new model...")
            return self._train_model()

    def _train_model(self):
        """Train the model from GLODAP data."""
        data_path = DATA_DIR / "GLODAPv2.2023_Merged_Master_File.csv"

        if not data_path.exists():
            print(f"Error: Data file not found at {data_path}")
            return False

        print(f"Loading {data_path.name}...")
        df_raw = pd.read_csv(data_path, na_values=[-9999, -999])

        cols = {
            'G2cruise': 'cruise', 'G2latitude': 'lat', 'G2longitude': 'lon',
            'G2depth': 'depth', 'G2theta': 'theta', 'G2salinity': 'salinity',
            'G2oxygen': 'oxygen', 'G2nitrate': 'nitrate', 'G2phosphate': 'phosphate',
            'G2silicate': 'silicate', 'G2c14': 'delta14c', 'G2c13': 'delta13c',
        }

        available = {k: v for k, v in cols.items() if k in df_raw.columns}
        df = df_raw[list(available.keys())].rename(columns=available)
        df['water_mass'] = df.apply(self._assign_water_mass, axis=1)

        deep_masses = ['NADW', 'AABW', 'AAIW', 'CDW']
        df_ml = df[
            (df['water_mass'].isin(deep_masses)) &
            (df['delta14c'].notna()) &
            (df['delta13c'].notna())
        ].copy()

        print(f"Training samples: {len(df_ml):,}")

        self.feature_cols_full = [
            'theta', 'salinity', 'depth', 'oxygen',
            'nitrate', 'phosphate', 'silicate', 'delta14c', 'delta13c'
        ]

        df_full = df_ml[self.feature_cols_full + ['water_mass']].dropna()
        X_full = df_full[self.feature_cols_full].values
        y_full = df_full['water_mass'].values

        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_full)

        self.scaler_full = StandardScaler()
        X_full_scaled = self.scaler_full.fit_transform(X_full)

        self.model_full = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0
        )
        self.model_full.fit(X_full_scaled, y_enc)

        X_isotope = df_ml[self.feature_cols_isotope].values
        y_isotope = df_ml['water_mass'].values
        y_isotope_enc = self.label_encoder.transform(y_isotope)

        self.scaler_isotope = StandardScaler()
        X_isotope_scaled = self.scaler_isotope.fit_transform(X_isotope)

        self.model_isotope = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='mlogloss', verbosity=0
        )
        self.model_isotope.fit(X_isotope_scaled, y_isotope_enc)

        self.training_data = df_ml[
            ['delta14c', 'delta13c', 'theta', 'salinity', 'depth', 'water_mass', 'lat', 'lon']
        ].dropna(subset=['delta14c', 'delta13c']).copy()

        self._fit_nn_model()

        joblib.dump({
            'model_full': self.model_full, 'model_isotope': self.model_isotope,
            'scaler_full': self.scaler_full, 'scaler_isotope': self.scaler_isotope,
            'label_encoder': self.label_encoder, 'training_data': self.training_data,
            'feature_cols_full': self.feature_cols_full,
        }, MODELS_DIR / "waterprint_model.joblib")

        return True

    def _fit_nn_model(self):
        X_nn = self.training_data[['delta14c', 'delta13c']].values
        X_nn_scaled = self.scaler_isotope.transform(X_nn)
        self.nn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
        self.nn_model.fit(X_nn_scaled)

    @staticmethod
    def _assign_water_mass(row):
        depth = row.get('depth', np.nan)
        lat = row.get('lat', np.nan)
        salinity = row.get('salinity', np.nan)
        theta = row.get('theta', np.nan)

        if pd.isna(depth) or pd.isna(lat):
            return np.nan
        if depth < 500:
            return 'SURFACE'
        if depth > 4000 and (pd.isna(theta) or theta < 2.0):
            return 'AABW'
        if 500 <= depth <= 1500 and lat < -20:
            if pd.isna(salinity) or salinity < 34.5:
                return 'AAIW'
        if lat < -40 and 1000 <= depth <= 4000:
            return 'CDW'
        if 1500 <= depth <= 4000 and lat > 0:
            if pd.isna(salinity) or salinity > 34.8:
                return 'NADW'
        return 'OTHER'

    def predict_single(self, delta14c, delta13c, use_full_model=False, **kwargs):
        if use_full_model and all(v is not None for v in kwargs.values()):
            X = np.array([[kwargs.get('theta'), kwargs.get('salinity'), kwargs.get('depth'),
                          kwargs.get('oxygen'), kwargs.get('nitrate'), kwargs.get('phosphate'),
                          kwargs.get('silicate'), delta14c, delta13c]])
            X_scaled = self.scaler_full.transform(X)
            proba = self.model_full.predict_proba(X_scaled)[0]
        else:
            X = np.array([[delta14c, delta13c]])
            X_scaled = self.scaler_isotope.transform(X)
            proba = self.model_isotope.predict_proba(X_scaled)[0]

        pred_idx = np.argmax(proba)
        pred_class = self.label_encoder.classes_[pred_idx]
        return pred_class, proba[pred_idx], {self.label_encoder.classes_[i]: float(proba[i]) for i in range(len(proba))}

    def predict_batch(self, df):
        results = []
        for idx, row in df.iterrows():
            delta14c = row.get('delta14c', row.get('d14c', None))
            delta13c = row.get('delta13c', row.get('d13c', None))

            if pd.isna(delta14c) or pd.isna(delta13c):
                results.append({'sample_id': idx, 'prediction': 'N/A', 'confidence': 0,
                               'NADW_prob': 0, 'AABW_prob': 0, 'AAIW_prob': 0, 'CDW_prob': 0})
                continue

            pred_class, confidence, proba_dict = self.predict_single(delta14c, delta13c)
            results.append({
                'sample_id': idx, 'delta14c': delta14c, 'delta13c': delta13c,
                'prediction': pred_class, 'confidence': confidence,
                'NADW_prob': proba_dict.get('NADW', 0), 'AABW_prob': proba_dict.get('AABW', 0),
                'AAIW_prob': proba_dict.get('AAIW', 0), 'CDW_prob': proba_dict.get('CDW', 0),
            })
        return pd.DataFrame(results)

    def find_similar_samples(self, delta14c, delta13c, n_neighbors=5):
        X = np.array([[delta14c, delta13c]])
        X_scaled = self.scaler_isotope.transform(X)
        distances, indices = self.nn_model.kneighbors(X_scaled, n_neighbors=n_neighbors)
        similar = self.training_data.iloc[indices[0]].copy()
        similar['distance'] = distances[0]
        similar['similarity'] = 1 / (1 + distances[0])
        return similar

    def get_uncertainty(self, delta14c, delta13c):
        X = np.array([[delta14c, delta13c]])
        X_scaled = self.scaler_isotope.transform(X)
        proba = self.model_isotope.predict_proba(X_scaled)[0]
        entropy = -np.sum(proba * np.log(proba + 1e-10))
        sorted_proba = np.sort(proba)[::-1]
        distances, _ = self.nn_model.kneighbors(X_scaled, n_neighbors=5)
        return {
            'entropy': entropy, 'normalized_entropy': entropy / np.log(len(proba)),
            'confidence_margin': sorted_proba[0] - sorted_proba[1],
            'mean_neighbor_distance': np.mean(distances),
            'is_low_confidence': proba.max() < 0.5,
            'is_near_boundary': (sorted_proba[0] - sorted_proba[1]) < 0.2,
            'is_outlier': np.mean(distances) > 2.0
        }


model = WaterPrintModel()


def initialize_model():
    return "Model loaded successfully" if model.load_or_train() else "Failed to load model"


def classify_single(delta14c, delta13c, theta, salinity, depth, oxygen, nitrate, phosphate, silicate, use_full):
    if delta14c is None or delta13c is None:
        return "Please provide Δ¹⁴C and δ¹³C values", None, None

    try:
        pred_class, confidence, proba_dict = model.predict_single(
            delta14c, delta13c, use_full_model=use_full,
            theta=theta, salinity=salinity, depth=depth, oxygen=oxygen,
            nitrate=nitrate, phosphate=phosphate, silicate=silicate
        )

        result = f"**Predicted Water Mass: {pred_class}**\n\nConfidence: {confidence:.1%}\n\n"
        result += f"{WATER_MASS_DESCRIPTIONS.get(pred_class, '')}\n\n"
        result += "**Probabilities:**\n"
        for wm, prob in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
            result += f"- {wm}: {prob:.1%}\n"

        # Probability plot
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        classes = list(proba_dict.keys())
        probs = [proba_dict[c] for c in classes]
        colors = [WATER_MASS_COLORS.get(c, 'gray') for c in classes]
        ax1.barh(classes, probs, color=colors)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Probability')
        ax1.set_title('Water Mass Classification')
        plt.tight_layout()

        # Isotope space plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        for wm in WATER_MASS_COLORS:
            mask = model.training_data['water_mass'] == wm
            if mask.sum() > 0:
                ax2.scatter(model.training_data.loc[mask, 'delta14c'],
                           model.training_data.loc[mask, 'delta13c'],
                           c=WATER_MASS_COLORS[wm], label=wm, alpha=0.3, s=10)
        ax2.scatter([delta14c], [delta13c], c='black', s=200, marker='*',
                   edgecolors='white', linewidths=2, zorder=10, label='Your sample')
        ax2.set_xlabel('Δ¹⁴C (‰)')
        ax2.set_ylabel('δ¹³C (‰)')
        ax2.set_title('Sample in Isotope Fingerprint Space')
        ax2.legend()
        plt.tight_layout()

        return result, fig1, fig2

    except Exception as e:
        return f"Error: {str(e)}", None, None


def classify_batch(file):
    if file is None:
        return "Please upload a CSV file", None, None

    try:
        df = pd.read_csv(file.name)

        # Detect columns
        for col in df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'delta14' in col_lower or 'd14c' in col_lower:
                df = df.rename(columns={col: 'delta14c'})
            elif 'delta13' in col_lower or 'd13c' in col_lower:
                df = df.rename(columns={col: 'delta13c'})

        if 'delta14c' not in df.columns or 'delta13c' not in df.columns:
            return "Could not find Δ¹⁴C and δ¹³C columns", None, None

        results = model.predict_batch(df)

        summary = f"**Total samples:** {len(results)}\n\n"
        summary += f"**Successfully classified:** {(results['prediction'] != 'N/A').sum()}\n\n"
        summary += "**Distribution:**\n"
        for wm, count in results['prediction'].value_counts().items():
            summary += f"- {wm}: {count}\n"

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        valid = results[results['prediction'] != 'N/A']
        counts = valid['prediction'].value_counts()
        colors = [WATER_MASS_COLORS.get(c, 'gray') for c in counts.index]
        axes[0].pie(counts.values, labels=counts.index, colors=colors, autopct='%1.1f%%')
        axes[0].set_title('Water Mass Distribution')

        for wm in WATER_MASS_COLORS:
            mask = valid['prediction'] == wm
            if mask.sum() > 0:
                axes[1].scatter(valid.loc[mask, 'delta14c'], valid.loc[mask, 'delta13c'],
                               c=WATER_MASS_COLORS[wm], label=wm, alpha=0.7, s=50)
        axes[1].set_xlabel('Δ¹⁴C (‰)')
        axes[1].set_ylabel('δ¹³C (‰)')
        axes[1].set_title('Classified Samples')
        axes[1].legend()
        plt.tight_layout()

        return summary, results, fig
    except Exception as e:
        return f"Error: {str(e)}", None, None


def show_isotope_space(highlight_wm):
    fig, ax = plt.subplots(figsize=(10, 8))
    for wm in WATER_MASS_COLORS:
        mask = model.training_data['water_mass'] == wm
        if mask.sum() > 0:
            alpha = 0.7 if wm == highlight_wm or highlight_wm == "All" else 0.15
            size = 25 if wm == highlight_wm or highlight_wm == "All" else 8
            ax.scatter(model.training_data.loc[mask, 'delta14c'],
                      model.training_data.loc[mask, 'delta13c'],
                      c=WATER_MASS_COLORS[wm], label=f"{wm} (n={mask.sum()})",
                      alpha=alpha, s=size)
    ax.set_xlabel('Δ¹⁴C (‰)')
    ax.set_ylabel('δ¹³C (‰)')
    ax.set_title('Isotope Fingerprint Space - GLODAP Training Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def show_ts_diagram(highlight_wm):
    fig, ax = plt.subplots(figsize=(10, 8))
    for wm in WATER_MASS_COLORS:
        mask = model.training_data['water_mass'] == wm
        if mask.sum() > 0:
            subset = model.training_data.loc[mask].dropna(subset=['theta', 'salinity'])
            if len(subset) > 0:
                alpha = 0.7 if wm == highlight_wm or highlight_wm == "All" else 0.15
                size = 25 if wm == highlight_wm or highlight_wm == "All" else 8
                ax.scatter(subset['salinity'], subset['theta'],
                          c=WATER_MASS_COLORS[wm], label=f"{wm} (n={len(subset)})",
                          alpha=alpha, s=size)
    ax.set_xlabel('Salinity (PSU)')
    ax.set_ylabel('Potential Temperature θ (°C)')
    ax.set_title('T-S Diagram - GLODAP Training Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def explain_prediction(delta14c, delta13c):
    if delta14c is None or delta13c is None:
        return "Please provide both values", None

    try:
        pred_class, confidence, proba_dict = model.predict_single(delta14c, delta13c)
        wm_stats = model.training_data.groupby('water_mass')[['delta14c', 'delta13c']].agg(['mean', 'std'])

        explanation = f"**Input:** Δ¹⁴C = {delta14c}‰, δ¹³C = {delta13c}‰\n\n"
        explanation += f"**Predicted:** {pred_class} (confidence: {confidence:.1%})\n\n"
        explanation += "**Distance to each water mass center:**\n\n"

        distances = []
        for wm in ['NADW', 'AABW', 'AAIW', 'CDW']:
            if wm in wm_stats.index:
                mean_d14c = wm_stats.loc[wm, ('delta14c', 'mean')]
                mean_d13c = wm_stats.loc[wm, ('delta13c', 'mean')]
                dist = np.sqrt((delta14c - mean_d14c)**2 + (delta13c - mean_d13c)**2)
                distances.append((wm, dist, mean_d14c, mean_d13c))
                marker = " ← Best Match" if wm == pred_class else ""
                explanation += f"- {wm}: {dist:.1f} (typical Δ¹⁴C: {mean_d14c:.1f}‰){marker}\n"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Position plot
        for wm in WATER_MASS_COLORS:
            mask = model.training_data['water_mass'] == wm
            if mask.sum() > 0:
                axes[0].scatter(model.training_data.loc[mask, 'delta14c'],
                               model.training_data.loc[mask, 'delta13c'],
                               c=WATER_MASS_COLORS[wm], label=wm, alpha=0.3, s=10)
        axes[0].scatter([delta14c], [delta13c], c='black', s=200, marker='*',
                       edgecolors='white', linewidths=2, zorder=10)
        axes[0].set_xlabel('Δ¹⁴C (‰)')
        axes[0].set_ylabel('δ¹³C (‰)')
        axes[0].set_title('Sample Position')
        axes[0].legend()

        # Distance plot
        wm_names = [d[0] for d in distances]
        dists = [d[1] for d in distances]
        colors = [WATER_MASS_COLORS[wm] for wm in wm_names]
        bars = axes[1].barh(wm_names, dists, color=colors)
        axes[1].set_xlabel('Distance to Centroid')
        axes[1].set_title('Distance to Each Water Mass')
        plt.tight_layout()

        return explanation, fig
    except Exception as e:
        return f"Error: {str(e)}", None


def create_sample_map(similar):
    """Create an interactive map showing similar sample locations."""
    valid_coords = similar.dropna(subset=['lat', 'lon'])
    if len(valid_coords) == 0:
        return None

    center_lat = valid_coords['lat'].mean()
    center_lon = valid_coords['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='CartoDB positron',
                   width='100%', height=500)

    wm_colors = {'NADW': 'red', 'AABW': 'blue', 'AAIW': 'green', 'CDW': 'purple'}

    for i, (_, row) in enumerate(valid_coords.iterrows(), 1):
        wm = row['water_mass']
        color = wm_colors.get(wm, 'gray')
        popup = f"Sample #{i}<br>Water Mass: {wm}<br>Δ¹⁴C: {row['delta14c']:.1f}‰<br>δ¹³C: {row['delta13c']:.2f}‰"

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8 + (row['similarity'] * 8),
            popup=popup, color=color, fill=True,
            fillColor=color, fillOpacity=0.7, weight=2
        ).add_to(m)

    return m


def find_similar(delta14c, delta13c, n_neighbors):
    if delta14c is None or delta13c is None:
        return "Please provide both values", None, "<p>No data</p>"

    try:
        similar = model.find_similar_samples(delta14c, delta13c, int(n_neighbors))

        result = f"**Your input:** Δ¹⁴C = {delta14c}‰, δ¹³C = {delta13c}‰\n\n"
        result += f"**Top {int(n_neighbors)} most similar samples:**\n\n"

        for i, (_, row) in enumerate(similar.iterrows(), 1):
            loc = f"{row.get('lat', 0):.1f}°, {row.get('lon', 0):.1f}°"
            result += f"{i}. {row['water_mass']} - Δ¹⁴C: {row['delta14c']:.1f}‰, δ¹³C: {row['delta13c']:.2f}‰, Depth: {row.get('depth', 0):.0f}m ({loc})\n"

        result += "\n**Distribution:**\n"
        for wm, count in similar['water_mass'].value_counts().items():
            result += f"- {wm}: {count} samples\n"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        for wm in WATER_MASS_COLORS:
            mask = model.training_data['water_mass'] == wm
            if mask.sum() > 0:
                ax.scatter(model.training_data.loc[mask, 'delta14c'],
                          model.training_data.loc[mask, 'delta13c'],
                          c=WATER_MASS_COLORS[wm], alpha=0.1, s=5)

        for wm in WATER_MASS_COLORS:
            mask = similar['water_mass'] == wm
            if mask.sum() > 0:
                ax.scatter(similar.loc[mask, 'delta14c'], similar.loc[mask, 'delta13c'],
                          c=WATER_MASS_COLORS[wm], label=wm, alpha=0.9, s=100,
                          edgecolors='black', linewidths=1)

        ax.scatter([delta14c], [delta13c], c='yellow', s=300, marker='*',
                  edgecolors='black', linewidths=2, zorder=10, label='Your sample')
        ax.set_xlabel('Δ¹⁴C (‰)')
        ax.set_ylabel('δ¹³C (‰)')
        ax.set_title(f'Your Sample and {int(n_neighbors)} Most Similar Samples')
        ax.legend()
        plt.tight_layout()

        # Map - generate with fixed height to match plot
        sample_map = create_sample_map(similar)
        if sample_map:
            # Get raw HTML and wrap with fixed height container
            raw_html = sample_map._repr_html_()
            # Replace the responsive padding with fixed height
            map_html = f'<div style="height:500px;width:100%;">{raw_html}</div>'
        else:
            map_html = "<p style='padding:20px;'>No coordinates available for map display.</p>"

        return result, fig, map_html
    except Exception as e:
        return f"Error: {str(e)}", None, "<p>Error</p>"


def quantify_uncertainty(delta14c, delta13c):
    if delta14c is None or delta13c is None:
        return "Please provide both values", None

    try:
        pred_class, confidence, proba_dict = model.predict_single(delta14c, delta13c)
        uncertainty = model.get_uncertainty(delta14c, delta13c)

        if uncertainty['normalized_entropy'] < 0.3:
            status = "LOW - Model is confident"
        elif uncertainty['normalized_entropy'] < 0.6:
            status = "MODERATE - Some ambiguity"
        else:
            status = "HIGH - Model is uncertain"

        result = f"**Input:** Δ¹⁴C = {delta14c}‰, δ¹³C = {delta13c}‰\n\n"
        result += f"**Predicted:** {pred_class} (confidence: {confidence:.1%})\n\n"
        result += f"**Uncertainty Level:** {status}\n\n"
        result += "**Metrics:**\n"
        result += f"- Normalized Entropy: {uncertainty['normalized_entropy']:.2f}\n"
        result += f"- Confidence Margin: {uncertainty['confidence_margin']:.2f}\n"
        result += f"- Distance to Neighbors: {uncertainty['mean_neighbor_distance']:.2f}\n\n"

        if uncertainty['is_near_boundary']:
            result += "⚠️ Sample is near classification boundary\n"
        if uncertainty['is_outlier']:
            result += "⚠️ Sample is far from training data\n"

        result += "\n**Probabilities:**\n"
        for wm, prob in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
            result += f"- {wm}: {prob:.1%}\n"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        classes = list(proba_dict.keys())
        probs = [proba_dict[c] for c in classes]
        colors = [WATER_MASS_COLORS.get(c, 'gray') for c in classes]
        axes[0].bar(classes, probs, color=colors, alpha=0.7, edgecolor='black')
        axes[0].axhline(y=0.5, color='red', linestyle='--', label='50%')
        axes[0].axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Random')
        axes[0].set_ylabel('Probability')
        axes[0].set_title('Classification Probabilities')
        axes[0].legend()
        axes[0].set_ylim(0, 1)

        for wm in WATER_MASS_COLORS:
            mask = model.training_data['water_mass'] == wm
            if mask.sum() > 0:
                axes[1].scatter(model.training_data.loc[mask, 'delta14c'],
                               model.training_data.loc[mask, 'delta13c'],
                               c=WATER_MASS_COLORS[wm], alpha=0.3, s=10, label=wm)
        axes[1].scatter([delta14c], [delta13c], c='black', s=200, marker='*',
                       edgecolors='white', linewidths=2, zorder=10)
        axes[1].set_xlabel('Δ¹⁴C (‰)')
        axes[1].set_ylabel('δ¹³C (‰)')
        axes[1].set_title('Sample Position')
        axes[1].legend()
        plt.tight_layout()

        return result, fig
    except Exception as e:
        return f"Error: {str(e)}", None


def create_interface():
    init_status = initialize_model()

    # Force light mode with JavaScript and minimal CSS
    force_light_js = """
    function() {
        // Remove dark mode classes
        document.body.classList.remove('dark');
        document.documentElement.classList.remove('dark');
        document.documentElement.style.colorScheme = 'light';
        // Prevent dark mode
        const observer = new MutationObserver(() => {
            document.body.classList.remove('dark');
            document.documentElement.classList.remove('dark');
        });
        observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    }
    """

    custom_css = """
    /* Force light mode */
    :root, html, body, .gradio-container, .dark {
        color-scheme: light only !important;
    }

    * {
        --neutral-950: #0a0a0a !important;
        --neutral-900: #171717 !important;
        --neutral-800: #262626 !important;
        --neutral-700: #404040 !important;
        --neutral-600: #525252 !important;
        --neutral-500: #737373 !important;
        --neutral-400: #a3a3a3 !important;
        --neutral-300: #d4d4d4 !important;
        --neutral-200: #e5e5e5 !important;
        --neutral-100: #f5f5f5 !important;
        --neutral-50: #fafafa !important;
    }

    html, body, .gradio-container, .dark, [data-theme="dark"] {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }

    .gradio-container {
        max-width: 1400px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* All text dark */
    p, span, label, h1, h2, h3, h4, h5, h6, div, a, li, td, th {
        color: #1a1a1a !important;
    }

    h1 {
        border-bottom: 2px solid #3182ce !important;
        padding-bottom: 0.5rem !important;
    }

    /* Input fields */
    input, textarea, select, .input-container, [data-testid="number-input"] {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border-color: #d1d5db !important;
    }

    /* Blocks and panels */
    .block, .panel, .form, .container, .wrap, .prose, .markdown {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* Tabs */
    .tabs, .tab-nav, .tabitem {
        background: #f9fafb !important;
    }
    button.tab-nav-button {
        color: #374151 !important;
        background: #e5e7eb !important;
    }
    button.tab-nav-button.selected {
        background: #3182ce !important;
        color: #ffffff !important;
    }

    /* Accordion */
    .accordion {
        background: #f9fafb !important;
        border-color: #e5e7eb !important;
    }

    /* Tables */
    table, tr, td, th, thead, tbody {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border-color: #e5e7eb !important;
    }
    th {
        background: #f3f4f6 !important;
    }

    /* Plots */
    .plot-container, .js-plotly-plot, .plotly {
        background: #ffffff !important;
    }

    /* Map */
    .map-container iframe {
        height: 500px !important;
        width: 100% !important;
    }

    /* Labels */
    .label-wrap, .label {
        color: #374151 !important;
    }

    /* Info text */
    .info {
        color: #6b7280 !important;
    }
    """

    with gr.Blocks(title="WaterPrint", theme=gr.themes.Soft(), css=custom_css, js=force_light_js) as demo:

        gr.Markdown("# WaterPrint: Water Mass Classification")
        gr.Markdown("Classify ocean water masses using isotopic fingerprints (Δ¹⁴C, δ¹³C) — no geographic coordinates required")
        gr.Markdown(f"**Status:** {init_status} | **Water Masses:** NADW, AABW, AAIW, CDW | **n = 7,138 samples**")

        with gr.Tabs():
            # Tab 1: Classify
            with gr.TabItem("Classify Sample"):
                with gr.Row():
                    with gr.Column(scale=1):
                        d14c = gr.Number(label="Δ¹⁴C (‰)", value=-120, info="Typical: -200 to 0")
                        d13c = gr.Number(label="δ¹³C (‰)", value=0.5, info="Typical: -1 to 2")
                        with gr.Accordion("Full Feature Model (optional)", open=False):
                            use_full = gr.Checkbox(label="Use all parameters", value=False)
                            theta = gr.Number(label="Temperature θ (°C)")
                            sal = gr.Number(label="Salinity (PSU)")
                            dep = gr.Number(label="Depth (m)")
                            oxy = gr.Number(label="Oxygen (μmol/kg)")
                            nit = gr.Number(label="Nitrate (μmol/kg)")
                            pho = gr.Number(label="Phosphate (μmol/kg)")
                            sil = gr.Number(label="Silicate (μmol/kg)")
                        btn1 = gr.Button("Classify", variant="primary")
                    with gr.Column(scale=2):
                        out1 = gr.Markdown()
                        plot1 = gr.Plot(label="Probabilities")
                        plot2 = gr.Plot(label="Isotope Space")
                btn1.click(classify_single, [d14c, d13c, theta, sal, dep, oxy, nit, pho, sil, use_full],
                          [out1, plot1, plot2])

            # Tab 2: Batch
            with gr.TabItem("Batch Classification"):
                gr.Markdown("Upload CSV with delta14c and delta13c columns")
                file_in = gr.File(label="Upload CSV", file_types=[".csv"])
                btn2 = gr.Button("Process", variant="primary")
                out2 = gr.Markdown()
                df_out = gr.Dataframe(label="Results")
                plot3 = gr.Plot()
                btn2.click(classify_batch, [file_in], [out2, df_out, plot3])

            # Tab 3: Visualization
            with gr.TabItem("Visualization"):
                highlight = gr.Dropdown(["All", "NADW", "AABW", "AAIW", "CDW"], value="All", label="Highlight")
                with gr.Row():
                    plot4 = gr.Plot(label="Isotope Space")
                    plot5 = gr.Plot(label="T-S Diagram")
                btn3a = gr.Button("Show Isotope Space")
                btn3b = gr.Button("Show T-S Diagram")
                btn3a.click(show_isotope_space, [highlight], [plot4])
                btn3b.click(show_ts_diagram, [highlight], [plot5])
                demo.load(show_isotope_space, [highlight], [plot4])

            # Tab 4: Explainability
            with gr.TabItem("Explainability"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ex_d14c = gr.Number(label="Δ¹⁴C (‰)", value=-120)
                        ex_d13c = gr.Number(label="δ¹³C (‰)", value=0.5)
                        btn4 = gr.Button("Explain", variant="primary")
                    with gr.Column(scale=2):
                        out4 = gr.Markdown()
                        plot6 = gr.Plot()
                btn4.click(explain_prediction, [ex_d14c, ex_d13c], [out4, plot6])

            # Tab 5: Similar Samples
            with gr.TabItem("Find Similar"):
                gr.Markdown("Find similar samples in GLODAP database")
                with gr.Row():
                    with gr.Column(scale=1):
                        sim_d14c = gr.Number(label="Δ¹⁴C (‰)", value=-120)
                        sim_d13c = gr.Number(label="δ¹³C (‰)", value=0.5)
                        n_sim = gr.Slider(3, 20, value=5, step=1, label="Number of samples")
                        btn5 = gr.Button("Find Similar", variant="primary")
                    with gr.Column(scale=2):
                        out5 = gr.Markdown()
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        plot7 = gr.Plot(label="Isotope Space")
                    with gr.Column(scale=1):
                        map_html = gr.HTML(label="Sample Locations")
                btn5.click(find_similar, [sim_d14c, sim_d13c, n_sim], [out5, plot7, map_html])

            # Tab 6: Uncertainty
            with gr.TabItem("Uncertainty"):
                with gr.Row():
                    with gr.Column(scale=1):
                        unc_d14c = gr.Number(label="Δ¹⁴C (‰)", value=-120)
                        unc_d13c = gr.Number(label="δ¹³C (‰)", value=0.5)
                        btn6 = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=2):
                        out6 = gr.Markdown()
                        plot8 = gr.Plot()
                btn6.click(quantify_uncertainty, [unc_d14c, unc_d13c], [out6, plot8])

        gr.Markdown("---")
        gr.Markdown("**WaterPrint** | Gruber (2025) | Data: GLODAP v2.2023 | Isotope-only accuracy: 74.1% ± 1.7% | LOCO: 70.4%")

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, ssr_mode=False)
