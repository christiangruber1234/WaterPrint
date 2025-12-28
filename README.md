---
title: WaterPrint
emoji: 🌊
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# WaterPrint

Machine learning classification of ocean water masses using radiocarbon and stable carbon isotopes.

## Overview

WaterPrint classifies major ocean water masses—North Atlantic Deep Water (NADW), Antarctic Bottom Water (AABW), Antarctic Intermediate Water (AAIW), and Circumpolar Deep Water (CDW)—using the GLODAP v2.2023 dataset.

## Key Finding

**Isotopes alone (Δ¹⁴C + δ¹³C) achieve 74.1% ± 1.7% classification accuracy**—nearly three times chance level—without explicit depth, latitude, or salinity coordinates as model inputs.

The 91‰ difference in mean Δ¹⁴C between NADW (−86‰) and AABW (−177‰) reflects approximately 870 years of ventilation age separation (Cohen's d = 2.2), encoding temporal information unavailable from physical properties.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete analysis pipeline (downloads data, runs analysis, generates figures):

```bash
python run.py
```

## Usage

### Download Data

```bash
python data/download_glodap.py
```

### Isotope Classification (Core Experiment)

```bash
python src/isotope_classification.py
```

### Full Analysis

```bash
python src/waterprint_analysis.py
```

### Generate Figures

```bash
python src/generate_figures.py
```

## Repository Structure

```
├── run.py                          # Complete pipeline
├── src/
│   ├── isotope_classification.py   # Core experiment: Δ¹⁴C + δ¹³C only
│   ├── waterprint_analysis.py      # Full-feature classification
│   └── generate_figures.py         # Reproduce manuscript figures
├── data/
│   ├── download_glodap.py          # Download GLODAP dataset
│   └── README.md                   # Data instructions
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite using the information in [CITATION.cff](CITATION.cff).
