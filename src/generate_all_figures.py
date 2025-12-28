#!/usr/bin/env python3
"""
WaterPrint: Unified Figure Generation Script
=============================================

Generates ALL figures for the manuscript and supplementary information
with a consistent Seaborn style and coolwarm pastel color scheme.

Color Logic (based on ventilation age - coolwarm gradient):
- NADW: Warm/coral (recently ventilated, North Atlantic)
- AAIW: Light teal (recently ventilated, Antarctic Intermediate)
- CDW:  Light blue (old, Circumpolar Deep Water)
- AABW: Slate blue (oldest, Antarctic Bottom Water)

Usage:
    python generate_all_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, LeaveOneGroupOut, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import xgboost as xgb

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "manuscript" / "revision_5" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# =============================================================================
# UNIFIED COLOR SCHEME - Coolwarm Pastel
# =============================================================================
# Logic: Ventilation age gradient from warm (young) to cool (old)
# Young waters (recently equilibrated with atmosphere) = warm colors
# Old waters (isolated from atmosphere) = cool colors

WATER_MASS_COLORS = {
    # Warm/young water masses (warm pastel colors)
    'NADW': '#E8998D',   # Coral pastel - North Atlantic, young
    'AAIW': '#98D4BB',   # Teal pastel - Antarctic Intermediate, young
    # Cool/old water masses (cool pastel colors)
    'CDW':  '#A8D5E5',   # Light blue pastel - Circumpolar Deep, old
    'AABW': '#8B9DC3',   # Slate blue pastel - Antarctic Bottom, oldest
}

# Consistent order: by ventilation age (young to old)
WATER_MASS_ORDER = ['NADW', 'AAIW', 'CDW', 'AABW']

# Additional colors for plots
ACCENT_COLORS = {
    'primary': '#5B7C99',      # Steel blue
    'secondary': '#D4A574',    # Warm tan
    'success': '#7CB07F',      # Soft green
    'warning': '#E8B87D',      # Soft orange
    'error': '#D4847C',        # Soft red
    'neutral': '#9E9E9E',      # Gray
}

# Coolwarm colormap for heatmaps
HEATMAP_CMAP = 'coolwarm'
SEQUENTIAL_CMAP = sns.light_palette('#5B7C99', as_cmap=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def download_glodap():
    """Download GLODAP dataset if not present."""
    import urllib.request
    import gzip
    import shutil

    csv_path = DATA_DIR / "GLODAPv2.2023_Merged_Master_File.csv"
    gz_path = DATA_DIR / "GLODAPv2.2023_Merged_Master_File.csv.gz"

    if csv_path.exists():
        return csv_path

    if gz_path.exists():
        print("Decompressing GLODAP data...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return csv_path

    print("Downloading GLODAP v2.2023 dataset...")
    url = "https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/GLODAPv2.2023_Merged_Master_File.csv"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, csv_path)
        print(f"  Downloaded to {csv_path}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Please download manually from: https://www.glodap.info/")
        raise

    return csv_path


def load_glodap():
    """Load GLODAP dataset with proper column mapping."""
    data_path = download_glodap()
    print(f"Loading {data_path.name}...")
    df_raw = pd.read_csv(data_path, na_values=[-9999, -999])

    cols = {
        'G2cruise': 'cruise',
        'G2year': 'year',
        'G2latitude': 'lat',
        'G2longitude': 'lon',
        'G2depth': 'depth',
        'G2theta': 'theta',
        'G2salinity': 'salinity',
        'G2oxygen': 'oxygen',
        'G2nitrate': 'nitrate',
        'G2phosphate': 'phosphate',
        'G2silicate': 'silicate',
        'G2c14': 'delta14c',
        'G2c13': 'delta13c',
    }

    df = df_raw[[c for c in cols.keys() if c in df_raw.columns]].rename(
        columns={k: v for k, v in cols.items() if k in df_raw.columns}
    )

    print(f"  Loaded {len(df):,} samples")
    return df


def assign_water_mass(row):
    """Assign water mass based on oceanographic criteria."""
    depth = row.get('depth', np.nan)
    lat = row.get('lat', np.nan)
    salinity = row.get('salinity', np.nan)
    theta = row.get('theta', np.nan)

    if pd.isna(depth) or pd.isna(lat):
        return np.nan

    if depth < 500:
        return 'SURFACE'

    # AABW: >4000m, very cold
    if depth > 4000 and (pd.isna(theta) or theta < 2.0):
        return 'AABW'

    # AAIW: 500-1500m, Southern Ocean, low salinity
    if 500 <= depth <= 1500 and lat < -20:
        if pd.isna(salinity) or salinity < 34.5:
            return 'AAIW'

    # CDW: 1000-4000m, high southern latitudes
    if lat < -40 and 1000 <= depth <= 4000:
        return 'CDW'

    # NADW: 1500-4000m, Northern Hemisphere, high salinity
    if 1500 <= depth <= 4000 and lat > 0:
        if pd.isna(salinity) or salinity > 34.8:
            return 'NADW'

    return 'OTHER'


def prepare_isotope_data(df, require_both=True):
    """Prepare dataset for isotope-only analysis."""
    df = df.copy()
    df['water_mass'] = df.apply(assign_water_mass, axis=1)

    deep_masses = ['NADW', 'AABW', 'AAIW', 'CDW']

    if require_both:
        df_filtered = df[
            (df['water_mass'].isin(deep_masses)) &
            (df['delta14c'].notna()) &
            (df['delta13c'].notna())
        ].copy()
    else:
        df_filtered = df[
            (df['water_mass'].isin(deep_masses)) &
            (df['delta14c'].notna())
        ].copy()

    return df_filtered


# =============================================================================
# FIGURE 1: ISOTOPE-ONLY CLASSIFICATION (Main Figure)
# =============================================================================

def figure_1_isotope_classification(df):
    """
    Figure 1: Isotope-only classification of ocean water masses.
    4 panels: A) Δ14C distributions, B) Isotope space, C) Confusion matrix, D) Accuracy comparison
    """
    print("\n" + "="*60)
    print("Creating Figure 1: Isotope-Only Classification")
    print("="*60)

    df_both = prepare_isotope_data(df, require_both=True)
    df_d14c = prepare_isotope_data(df, require_both=False)

    # Prepare data
    X_both = df_both[['delta14c', 'delta13c']].values
    X_d14c = df_d14c[['delta14c']].values
    y_both = df_both['water_mass'].values
    y_d14c = df_d14c['water_mass'].values

    le = LabelEncoder()
    y_both_enc = le.fit_transform(y_both)

    le_d14c = LabelEncoder()
    y_d14c_enc = le_d14c.fit_transform(y_d14c)

    # Cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Δ14C only accuracy
    acc_d14c = []
    for train_idx, test_idx in skf.split(X_d14c, y_d14c_enc):
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_d14c[train_idx], y_d14c_enc[train_idx])
        acc_d14c.append(accuracy_score(y_d14c_enc[test_idx], model.predict(X_d14c[test_idx])))

    # Δ14C + δ13C accuracy
    acc_both = []
    for train_idx, test_idx in skf.split(X_both, y_both_enc):
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_both[train_idx], y_both_enc[train_idx])
        acc_both.append(accuracy_score(y_both_enc[test_idx], model.predict(X_both[test_idx])))

    # Final model for confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X_both, y_both_enc, test_size=0.2, random_state=42, stratify=y_both_enc
    )
    model_final = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A) Δ14C Distribution by Water Mass
    ax1 = axes[0, 0]
    for wm in WATER_MASS_ORDER:
        data = df_d14c[df_d14c['water_mass'] == wm]['delta14c']
        sns.histplot(data, bins=30, alpha=0.6, label=f"{wm} (n={len(data):,})",
                    color=WATER_MASS_COLORS[wm], ax=ax1, kde=False, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel(r'$\Delta^{14}$C (‰)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('A) $\\Delta^{14}$C Distribution by Water Mass', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)

    # B) Isotope Fingerprint Space
    ax2 = axes[0, 1]
    for wm in WATER_MASS_ORDER:
        mask = df_both['water_mass'] == wm
        ax2.scatter(df_both.loc[mask, 'delta14c'], df_both.loc[mask, 'delta13c'],
                   c=WATER_MASS_COLORS[wm], label=wm, alpha=0.6, s=20, edgecolor='white', linewidth=0.3)

    ax2.set_xlabel(r'$\Delta^{14}$C (‰)', fontsize=12)
    ax2.set_ylabel(r'$\delta^{13}$C (‰)', fontsize=12)
    ax2.set_title('B) Isotope Fingerprint Space', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)

    # C) Confusion Matrix
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)

    # Use ordered class names
    class_order = [le.transform([wm])[0] for wm in WATER_MASS_ORDER if wm in le.classes_]
    cm_ordered = cm[np.ix_(class_order, class_order)]

    sns.heatmap(cm_ordered, annot=True, fmt='d', cmap=SEQUENTIAL_CMAP,
                xticklabels=WATER_MASS_ORDER, yticklabels=WATER_MASS_ORDER,
                ax=ax3, cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='white')
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('True', fontsize=12)
    ax3.set_title(f'C) Confusion Matrix (Isotopes Only)\nAccuracy: {np.mean(acc_both):.1%}',
                  fontsize=13, fontweight='bold')

    # D) Accuracy Comparison
    ax4 = axes[1, 1]

    feature_sets = [r'$\Delta^{14}$C only', r'$\Delta^{14}$C + $\delta^{13}$C']
    accuracies = [np.mean(acc_d14c) * 100, np.mean(acc_both) * 100]
    errors = [np.std(acc_d14c) * 100, np.std(acc_both) * 100]
    colors_bar = [ACCENT_COLORS['primary'], ACCENT_COLORS['success']]

    bars = ax4.bar(feature_sets, accuracies, yerr=errors, capsize=8,
                   color=colors_bar, edgecolor='white', linewidth=1.5, error_kw={'linewidth': 2})

    # Add chance level line
    ax4.axhline(y=25, color=ACCENT_COLORS['error'], linestyle='--', linewidth=2,
                label='Chance Level (25%)')

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('D) Classification Accuracy by Feature Set', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'fig_1_isotope_classification.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_1_isotope_classification.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_1_isotope_classification.png'}")
    plt.close()

    return {'acc_d14c': acc_d14c, 'acc_both': acc_both}


# =============================================================================
# FIGURE 2: RADIOCARBON SIGNATURES (Main Figure)
# =============================================================================

def figure_2_radiocarbon_signatures(df):
    """
    Figure 2: Radiocarbon signatures encode ventilation age.
    3 panels: A) T-S diagram, B) Δ14C vs depth, C) Boxplots with effect sizes
    """
    print("\n" + "="*60)
    print("Creating Figure 2: Radiocarbon Signatures")
    print("="*60)

    df_iso = prepare_isotope_data(df, require_both=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A) T-S Diagram
    ax1 = axes[0]
    for wm in WATER_MASS_ORDER:
        mask = df_iso['water_mass'] == wm
        if mask.sum() > 0:
            ax1.scatter(df_iso.loc[mask, 'salinity'], df_iso.loc[mask, 'theta'],
                       c=WATER_MASS_COLORS[wm], label=wm, alpha=0.6, s=20,
                       edgecolor='white', linewidth=0.3)

    ax1.set_xlabel('Salinity (PSU)', fontsize=12)
    ax1.set_ylabel('Potential Temperature (°C)', fontsize=12)
    ax1.set_title('A) Temperature-Salinity Diagram', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)

    # B) Δ14C vs Depth
    ax2 = axes[1]
    for wm in WATER_MASS_ORDER:
        mask = df_iso['water_mass'] == wm
        if mask.sum() > 0:
            ax2.scatter(df_iso.loc[mask, 'delta14c'], df_iso.loc[mask, 'depth'],
                       c=WATER_MASS_COLORS[wm], label=wm, alpha=0.6, s=20,
                       edgecolor='white', linewidth=0.3)

    ax2.invert_yaxis()
    ax2.set_xlabel(r'$\Delta^{14}$C (‰)', fontsize=12)
    ax2.set_ylabel('Depth (m)', fontsize=12)
    ax2.set_title(r'B) $\Delta^{14}$C vs Depth', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)

    # C) Boxplots with effect sizes
    ax3 = axes[2]

    # Prepare data for boxplot
    plot_data = []
    for wm in WATER_MASS_ORDER:
        values = df_iso[df_iso['water_mass'] == wm]['delta14c'].values
        for v in values:
            plot_data.append({'Water Mass': wm, 'Δ14C': v})
    plot_df = pd.DataFrame(plot_data)

    # Create boxplot with seaborn
    palette = [WATER_MASS_COLORS[wm] for wm in WATER_MASS_ORDER]
    sns.boxplot(data=plot_df, x='Water Mass', y='Δ14C', order=WATER_MASS_ORDER,
                palette=palette, ax=ax3, linewidth=1.5, fliersize=3)

    ax3.set_xlabel('Water Mass', fontsize=12)
    ax3.set_ylabel(r'$\Delta^{14}$C (‰)', fontsize=12)
    ax3.set_title(r'C) $\Delta^{14}$C Distributions', fontsize=13, fontweight='bold')

    # Add effect size annotations
    means = {wm: df_iso[df_iso['water_mass'] == wm]['delta14c'].mean() for wm in WATER_MASS_ORDER}
    stds = {wm: df_iso[df_iso['water_mass'] == wm]['delta14c'].std() for wm in WATER_MASS_ORDER}

    # Cohen's d for NADW vs AABW
    d_nadw_aabw = (means['NADW'] - means['AABW']) / np.sqrt((stds['NADW']**2 + stds['AABW']**2) / 2)

    ax3.annotate(f"Cohen's d = {d_nadw_aabw:.1f}\n(NADW vs AABW)",
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'fig_2_radiocarbon_signatures.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_2_radiocarbon_signatures.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_2_radiocarbon_signatures.png'}")
    plt.close()


# =============================================================================
# FIGURE 3: FEATURE IMPORTANCE (Main Figure)
# =============================================================================

def figure_3_feature_importance(df):
    """
    Figure 3: SHAP-style feature importance analysis.
    """
    print("\n" + "="*60)
    print("Creating Figure 3: Feature Importance")
    print("="*60)

    df_full = df.copy()
    df_full['water_mass'] = df_full.apply(assign_water_mass, axis=1)

    deep_masses = ['NADW', 'AABW', 'AAIW', 'CDW']

    features = ['depth', 'theta', 'salinity', 'oxygen', 'delta14c', 'delta13c',
                'nitrate', 'phosphate', 'silicate']

    df_complete = df_full[
        (df_full['water_mass'].isin(deep_masses)) &
        (df_full[features].notna().all(axis=1))
    ].copy()

    if len(df_complete) < 100:
        print("  Warning: Not enough complete samples for full-feature analysis")
        return

    X = df_complete[features].values
    y = df_complete['water_mass'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train model
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X, y_enc)

    # Get feature importance
    importance = model.feature_importances_

    # Sort by importance
    sorted_idx = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    # Create nicer feature names
    feature_labels = {
        'depth': 'Depth',
        'theta': 'Pot. Temperature (θ)',
        'salinity': 'Salinity',
        'oxygen': 'Dissolved Oxygen',
        'delta14c': r'$\Delta^{14}$C',
        'delta13c': r'$\delta^{13}$C',
        'nitrate': 'Nitrate',
        'phosphate': 'Phosphate',
        'silicate': 'Silicate',
    }
    sorted_labels = [feature_labels.get(f, f) for f in sorted_features]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color bars: highlight isotopes
    colors = []
    for f in sorted_features:
        if f in ['delta14c', 'delta13c']:
            colors.append(ACCENT_COLORS['success'])
        elif f in ['depth', 'salinity', 'theta']:
            colors.append(ACCENT_COLORS['warning'])
        else:
            colors.append(ACCENT_COLORS['primary'])

    bars = ax.barh(sorted_labels, sorted_importance, color=colors,
                   edgecolor='white', linewidth=1)

    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Feature Importance for Water Mass Classification', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT_COLORS['warning'], label='Label-defining features'),
        Patch(facecolor=ACCENT_COLORS['success'], label='Isotopic tracers'),
        Patch(facecolor=ACCENT_COLORS['primary'], label='Other features'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Add annotation
    ax.annotate(f"$\\Delta^{{14}}$C ranks among top features\ndespite not being used in label definitions",
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'fig_3_feature_importance.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_3_feature_importance.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_3_feature_importance.png'}")
    plt.close()


# =============================================================================
# SUPPLEMENTARY FIGURE S1: LODO Cross-Validation
# =============================================================================

def figure_s1_lodo_barplot(df):
    """
    Supplementary Figure S1: Leave-One-Decade-Out cross-validation.
    """
    print("\n" + "="*60)
    print("Creating Figure S1: LODO Barplot")
    print("="*60)

    df_iso = prepare_isotope_data(df, require_both=True)
    df_iso = df_iso[df_iso['year'].notna()].copy()
    df_iso['decade'] = (df_iso['year'] // 10) * 10

    X = df_iso[['delta14c', 'delta13c']].values
    y = df_iso['water_mass'].values
    decades = df_iso['decade'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    results = []
    for test_decade in sorted(df_iso['decade'].unique()):
        train_mask = decades != test_decade
        test_mask = decades == test_decade

        if test_mask.sum() < 50:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_enc[train_mask], y_enc[test_mask]

        if len(np.unique(y_train)) < 4 or len(np.unique(y_test)) < 2:
            continue

        model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'decade': f"{int(test_decade)}s",
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'n_samples': test_mask.sum()
        })

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    decades_labels = [r['decade'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    n_samples = [r['n_samples'] for r in results]

    # Gradient colors based on performance
    colors = [ACCENT_COLORS['error'] if a < 50 else
              ACCENT_COLORS['success'] if a > 70 else
              ACCENT_COLORS['warning'] for a in accuracies]

    bars = ax.bar(decades_labels, accuracies, color=colors,
                  edgecolor='white', linewidth=1.5)

    # Add sample size labels
    for bar, n in zip(bars, n_samples):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'n={n:,}', ha='center', va='bottom', fontsize=9)

    # Reference lines
    ax.axhline(y=74.1, color=ACCENT_COLORS['primary'], linestyle='--', linewidth=2,
               label='10-Fold CV (74.1%)')
    ax.axhline(y=25, color=ACCENT_COLORS['neutral'], linestyle=':', linewidth=1.5,
               label='Chance level (25%)')

    ax.set_xlabel('Test Decade', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Leave-One-Decade-Out Cross-Validation', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 90)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'fig_s1_lodo_barplot.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_s1_lodo_barplot.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_s1_lodo_barplot.png'}")
    plt.close()


# =============================================================================
# SUPPLEMENTARY FIGURE S2: Baseline Comparison
# =============================================================================

def figure_s2_baseline_comparison(df):
    """
    Supplementary Figure S2: Baseline comparison barplot.
    """
    print("\n" + "="*60)
    print("Creating Figure S2: Baseline Comparison")
    print("="*60)

    df_iso = prepare_isotope_data(df, require_both=True)

    X_both = df_iso[['delta14c', 'delta13c']].values
    X_d14c = df_iso[['delta14c']].values
    y = df_iso['water_mass'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    models = [
        ('Stratified\nRandom', DummyClassifier(strategy='stratified'), X_d14c),
        ('Majority\nClass', DummyClassifier(strategy='most_frequent'), X_d14c),
        (r'$\Delta^{14}$C only' + '\nLogReg', LogisticRegression(max_iter=1000, random_state=42), X_d14c),
        (r'$\Delta^{14}$C only' + '\nXGBoost', xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0), X_d14c),
        (r'$\Delta^{14}$C+$\delta^{13}$C' + '\nLogReg', LogisticRegression(max_iter=1000, random_state=42), X_both),
        (r'$\Delta^{14}$C+$\delta^{13}$C' + '\nXGBoost', xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0), X_both),
    ]

    results = []
    for name, model, X in models:
        accs = cross_val_score(model, X, y_enc, cv=skf, scoring='accuracy')
        results.append({
            'name': name,
            'accuracy': accs.mean() * 100,
            'std': accs.std() * 100
        })

    # Create figure
    fig, ax = plt.subplots(figsize=(11, 5))

    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    stds = [r['std'] for r in results]

    # Color gradient
    colors = [ACCENT_COLORS['error'], ACCENT_COLORS['error'],
              ACCENT_COLORS['warning'], ACCENT_COLORS['warning'],
              ACCENT_COLORS['success'], ACCENT_COLORS['success']]

    bars = ax.bar(names, accuracies, yerr=stds, capsize=5,
                  color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Baseline Comparison: Isotope-Only Classification', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 90)

    # Category labels
    ax.axvline(x=1.5, color=ACCENT_COLORS['neutral'], linestyle='--', alpha=0.5)
    ax.axvline(x=3.5, color=ACCENT_COLORS['neutral'], linestyle='--', alpha=0.5)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'fig_s2_baseline_comparison.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_s2_baseline_comparison.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_s2_baseline_comparison.png'}")
    plt.close()


# =============================================================================
# SUPPLEMENTARY FIGURE S3: Class-Weighted Recall
# =============================================================================

def figure_s3_class_weighted_recall(df):
    """
    Supplementary Figure S3: Class-weighted vs unweighted recall comparison.
    """
    print("\n" + "="*60)
    print("Creating Figure S3: Class-Weighted Recall")
    print("="*60)

    df_iso = prepare_isotope_data(df, require_both=True)

    X = df_iso[['delta14c', 'delta13c']].values
    y = df_iso['water_mass'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Calculate class weights
    class_counts = pd.Series(y).value_counts()
    total = len(y)
    n_classes = len(class_counts)

    sample_weights = np.array([
        total / (n_classes * class_counts[le.classes_[yi]])
        for yi in y_enc
    ])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    recalls = {'Unweighted': {wm: [] for wm in le.classes_},
               'Class-weighted': {wm: [] for wm in le.classes_}}

    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        sw_train = sample_weights[train_idx]

        # Unweighted
        model_unw = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        model_unw.fit(X_train, y_train)
        y_pred_unw = model_unw.predict(X_test)

        # Class-weighted
        model_w = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        model_w.fit(X_train, y_train, sample_weight=sw_train)
        y_pred_w = model_w.predict(X_test)

        for i, wm in enumerate(le.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                recalls['Unweighted'][wm].append((y_pred_unw[mask] == i).mean())
                recalls['Class-weighted'][wm].append((y_pred_w[mask] == i).mean())

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(WATER_MASS_ORDER))
    width = 0.35

    unweighted_means = [np.mean(recalls['Unweighted'].get(wm, [0])) for wm in WATER_MASS_ORDER]
    weighted_means = [np.mean(recalls['Class-weighted'].get(wm, [0])) for wm in WATER_MASS_ORDER]
    unweighted_stds = [np.std(recalls['Unweighted'].get(wm, [0])) for wm in WATER_MASS_ORDER]
    weighted_stds = [np.std(recalls['Class-weighted'].get(wm, [0])) for wm in WATER_MASS_ORDER]

    bars1 = ax.bar(x - width/2, unweighted_means, width, yerr=unweighted_stds,
                   label='Unweighted', color=ACCENT_COLORS['primary'],
                   edgecolor='white', capsize=4)
    bars2 = ax.bar(x + width/2, weighted_means, width, yerr=weighted_stds,
                   label='Class-weighted', color=ACCENT_COLORS['secondary'],
                   edgecolor='white', capsize=4)

    # Add value labels
    for bar, val in zip(bars1, unweighted_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, weighted_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Recall', fontsize=12)
    ax.set_xlabel('Water Mass', fontsize=12)
    ax.set_title('Effect of Class Weighting on Per-Class Recall', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(WATER_MASS_ORDER, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=10)

    # Add water mass color bands
    for i, wm in enumerate(WATER_MASS_ORDER):
        ax.axvspan(i - 0.45, i + 0.45, alpha=0.15, color=WATER_MASS_COLORS[wm])

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'fig_s3_class_weighted_recall.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_s3_class_weighted_recall.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_s3_class_weighted_recall.png'}")
    plt.close()


# =============================================================================
# SUPPLEMENTARY FIGURE S4: OTHER Confidence
# =============================================================================

def figure_s4_other_confidence(df):
    """
    Supplementary Figure S4: Classification confidence for OTHER samples.
    """
    print("\n" + "="*60)
    print("Creating Figure S4: OTHER Confidence")
    print("="*60)

    df_full = df.copy()
    df_full['water_mass'] = df_full.apply(assign_water_mass, axis=1)

    deep_masses = ['NADW', 'AABW', 'AAIW', 'CDW']

    # Training data
    df_train = df_full[
        (df_full['water_mass'].isin(deep_masses)) &
        (df_full['delta14c'].notna()) &
        (df_full['delta13c'].notna())
    ].copy()

    # OTHER samples
    df_other = df_full[
        (df_full['water_mass'] == 'OTHER') &
        (df_full['delta14c'].notna()) &
        (df_full['delta13c'].notna())
    ].copy()

    if len(df_other) < 10:
        print("  Warning: Not enough OTHER samples")
        return

    X_train = df_train[['delta14c', 'delta13c']].values
    y_train = df_train['water_mass'].values
    X_other = df_other[['delta14c', 'delta13c']].values

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train_enc)

    # Get probabilities
    probs_train = model.predict_proba(X_train)
    probs_other = model.predict_proba(X_other)

    max_probs_train = probs_train.max(axis=1)
    max_probs_other = probs_other.max(axis=1)

    predictions_other = le.classes_[model.predict(X_other)]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # A) Confidence histograms
    ax1 = axes[0]
    bins = np.linspace(0.25, 1.0, 31)

    ax1.hist(max_probs_train, bins=bins, alpha=0.6,
             label=f'Training (n={len(max_probs_train):,})',
             color=ACCENT_COLORS['primary'], edgecolor='white', linewidth=0.5)
    ax1.hist(max_probs_other, bins=bins, alpha=0.6,
             label=f'OTHER (n={len(max_probs_other):,})',
             color=ACCENT_COLORS['secondary'], edgecolor='white', linewidth=0.5)

    ax1.axvline(max_probs_train.mean(), color=ACCENT_COLORS['primary'],
                linestyle='--', linewidth=2,
                label=f'Training mean: {max_probs_train.mean():.2f}')
    ax1.axvline(max_probs_other.mean(), color=ACCENT_COLORS['secondary'],
                linestyle='--', linewidth=2,
                label=f'OTHER mean: {max_probs_other.mean():.2f}')

    ax1.set_xlabel('Maximum Classification Probability', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('A) Confidence Distribution', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(0.25, 1.0)

    # B) Predicted classes for OTHER
    ax2 = axes[1]
    pred_counts = pd.Series(predictions_other).value_counts()

    colors_pie = [WATER_MASS_COLORS.get(wm, ACCENT_COLORS['neutral']) for wm in pred_counts.index]
    wedges, texts, autotexts = ax2.pie(pred_counts.values, labels=pred_counts.index,
                                        autopct='%1.1f%%', colors=colors_pie,
                                        explode=[0.02]*len(pred_counts),
                                        textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax2.set_title('B) Predicted Classes for OTHER Samples', fontsize=13, fontweight='bold')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'fig_s4_other_confidence.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_s4_other_confidence.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_s4_other_confidence.png'}")
    plt.close()


# =============================================================================
# SUPPLEMENTARY FIGURE S5: NADW Sensitivity
# =============================================================================

def figure_s5_nadw_sensitivity(df):
    """
    Supplementary Figure S5: NADW geographic restriction sensitivity.
    """
    print("\n" + "="*60)
    print("Creating Figure S5: NADW Sensitivity")
    print("="*60)

    def assign_wm_original(row):
        depth, lat, salinity, theta = row.get('depth'), row.get('lat'), row.get('salinity'), row.get('theta')
        if pd.isna(depth) or pd.isna(lat): return np.nan
        if depth < 500: return 'SURFACE'
        if depth > 4000 and (pd.isna(theta) or theta < 2.0): return 'AABW'
        if 500 <= depth <= 1500 and lat < -20:
            if pd.isna(salinity) or salinity < 34.5: return 'AAIW'
        if lat < -40 and 1000 <= depth <= 4000: return 'CDW'
        if 1500 <= depth <= 4000 and lat > 0:
            if pd.isna(salinity) or salinity > 34.8: return 'NADW'
        return 'OTHER'

    def assign_wm_extended(row):
        depth, lat, salinity, theta = row.get('depth'), row.get('lat'), row.get('salinity'), row.get('theta')
        if pd.isna(depth) or pd.isna(lat): return np.nan
        if depth < 500: return 'SURFACE'
        if depth > 4000 and (pd.isna(theta) or theta < 2.0): return 'AABW'
        if 500 <= depth <= 1500 and lat < -20:
            if pd.isna(salinity) or salinity < 34.5: return 'AAIW'
        if lat < -40 and 1000 <= depth <= 4000: return 'CDW'
        if 1500 <= depth <= 4000 and lat > -30:  # Extended
            if pd.isna(salinity) or salinity > 34.8: return 'NADW'
        return 'OTHER'

    deep_masses = ['NADW', 'AABW', 'AAIW', 'CDW']
    results = {}

    for name, assign_func in [("Original\n(>0°N)", assign_wm_original),
                               ("Extended\n(>-30°S)", assign_wm_extended)]:
        df_temp = df.copy()
        df_temp['water_mass'] = df_temp.apply(assign_func, axis=1)

        df_iso = df_temp[
            (df_temp['water_mass'].isin(deep_masses)) &
            (df_temp['delta14c'].notna()) &
            (df_temp['delta13c'].notna())
        ].copy()

        X = df_iso[['delta14c', 'delta13c']].values
        y = df_iso['water_mass'].values
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        accs, bal_accs = [], []

        for train_idx, test_idx in skf.split(X, y_enc):
            model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
            model.fit(X[train_idx], y_enc[train_idx])
            y_pred = model.predict(X[test_idx])
            accs.append(accuracy_score(y_enc[test_idx], y_pred))
            bal_accs.append(balanced_accuracy_score(y_enc[test_idx], y_pred))

        results[name] = {
            'accuracy': np.mean(accs) * 100,
            'acc_std': np.std(accs) * 100,
            'balanced': np.mean(bal_accs) * 100,
            'bal_std': np.std(bal_accs) * 100,
            'n_nadw': len(df_iso[df_iso['water_mass'] == 'NADW']),
        }

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # A) Accuracy comparison
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35

    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    acc_stds = [results[n]['acc_std'] for n in names]
    bals = [results[n]['balanced'] for n in names]
    bal_stds = [results[n]['bal_std'] for n in names]

    bars1 = ax1.bar(x - width/2, accs, width, yerr=acc_stds, label='Accuracy',
                    color=ACCENT_COLORS['primary'], edgecolor='white', capsize=5)
    bars2 = ax1.bar(x + width/2, bals, width, yerr=bal_stds, label='Balanced Acc.',
                    color=ACCENT_COLORS['success'], edgecolor='white', capsize=5)

    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, bals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Performance (%)', fontsize=12)
    ax1.set_title('A) Classification Performance', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylim(0, 90)
    ax1.legend(loc='lower right', fontsize=10)

    # B) NADW sample count
    ax2 = axes[1]
    n_nadw = [results[n]['n_nadw'] for n in names]

    bars = ax2.bar(names, n_nadw,
                   color=[WATER_MASS_COLORS['NADW'], '#D4A574'],
                   edgecolor='white', linewidth=1.5)

    for bar, n in zip(bars, n_nadw):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{n:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Number of NADW Samples', fontsize=12)
    ax2.set_title('B) NADW Sample Count', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(n_nadw) * 1.2)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'fig_s5_nadw_sensitivity.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_s5_nadw_sensitivity.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_s5_nadw_sensitivity.png'}")
    plt.close()


# =============================================================================
# SUPPLEMENTARY FIGURE S6: LOCO Distribution
# =============================================================================

def figure_s6_loco_distribution(df):
    """
    Supplementary Figure S6: Leave-One-Cruise-Out accuracy distribution.
    """
    print("\n" + "="*60)
    print("Creating Figure S6: LOCO Distribution")
    print("="*60)

    df_iso = prepare_isotope_data(df, require_both=True)
    df_iso = df_iso[df_iso['cruise'].notna()].copy()

    X = df_iso[['delta14c', 'delta13c']].values
    y = df_iso['water_mass'].values
    groups = df_iso['cruise'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    logo = LeaveOneGroupOut()
    loco_accuracies = []

    for train_idx, test_idx in logo.split(X, y_enc, groups):
        if len(test_idx) < 5:
            continue
        if len(np.unique(y_enc[train_idx])) < 2 or len(np.unique(y_enc[test_idx])) < 2:
            continue

        model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X[train_idx], y_enc[train_idx])
        y_pred = model.predict(X[test_idx])
        loco_accuracies.append(accuracy_score(y_enc[test_idx], y_pred) * 100)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # A) Histogram
    ax1 = axes[0]
    sns.histplot(loco_accuracies, bins=20, color=ACCENT_COLORS['primary'],
                 edgecolor='white', ax=ax1, kde=True)

    ax1.axvline(np.mean(loco_accuracies), color=ACCENT_COLORS['error'],
                linestyle='--', linewidth=2, label=f'Mean: {np.mean(loco_accuracies):.1f}%')
    ax1.axvline(np.median(loco_accuracies), color=ACCENT_COLORS['success'],
                linestyle='--', linewidth=2, label=f'Median: {np.median(loco_accuracies):.1f}%')

    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('A) LOCO Accuracy Distribution', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)

    # B) Boxplot with individual points
    ax2 = axes[1]

    bp = ax2.boxplot([loco_accuracies], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=ACCENT_COLORS['primary'], alpha=0.6),
                     medianprops=dict(color=ACCENT_COLORS['error'], linewidth=2))

    # Add jittered points
    x_jitter = np.random.normal(1, 0.04, size=len(loco_accuracies))
    ax2.scatter(x_jitter, loco_accuracies, alpha=0.5, color=ACCENT_COLORS['secondary'],
                s=30, edgecolor='white', linewidth=0.5)

    ax2.axhline(25, color=ACCENT_COLORS['neutral'], linestyle=':', linewidth=1.5,
                label='Chance (25%)')
    ax2.axhline(74.1, color=ACCENT_COLORS['success'], linestyle='--', linewidth=2,
                label='10-Fold CV (74.1%)')

    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xticklabels(['LOCO'], fontsize=11)
    ax2.set_title(f'B) LOCO Summary (n={len(loco_accuracies)} cruises)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 105)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'fig_s6_loco_distribution.png', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_s6_loco_distribution.pdf')
    print(f"  Saved: {OUTPUT_DIR / 'fig_s6_loco_distribution.png'}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all figures."""
    print("="*70)
    print("WATERPRINT: UNIFIED FIGURE GENERATION")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Color scheme: Coolwarm Pastel")
    print("="*70)

    # Load data
    df = load_glodap()

    # Main figures
    print("\n" + "="*70)
    print("MAIN FIGURES")
    print("="*70)

    figure_1_isotope_classification(df)
    figure_2_radiocarbon_signatures(df)
    figure_3_feature_importance(df)

    # Supplementary figures
    print("\n" + "="*70)
    print("SUPPLEMENTARY FIGURES")
    print("="*70)

    figure_s1_lodo_barplot(df)
    figure_s2_baseline_comparison(df)
    figure_s3_class_weighted_recall(df)
    figure_s4_other_confidence(df)
    figure_s5_nadw_sensitivity(df)
    figure_s6_loco_distribution(df)

    # Summary
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("\nGenerated figures:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")

    print("\n" + "="*70)
    print("COLOR SCHEME REFERENCE")
    print("="*70)
    print("\nWater Mass Colors (Ventilation Age Gradient):")
    for wm in WATER_MASS_ORDER:
        print(f"  {wm}: {WATER_MASS_COLORS[wm]}")
    print("\nAccent Colors:")
    for name, color in ACCENT_COLORS.items():
        print(f"  {name}: {color}")


if __name__ == "__main__":
    main()
