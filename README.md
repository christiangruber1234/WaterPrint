---
title: WaterPrint
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
---

# WaterPrint

**Isotopic fingerprints classify ocean water masses without geographic coordinates**

Machine learning classification of ocean water masses using radiocarbon and stable carbon isotopes.

**Try it:** [Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/user5761/waterprint)

---

## Abstract

Identifying ocean water masses traditionally requires geographic context and physical properties. Here we demonstrate that isotopic tracers alone—radiocarbon (Δ¹⁴C) and stable carbon isotopes (δ¹³C)—can classify major deep water masses with 74.1% accuracy using machine learning, without explicit depth, latitude, or salinity inputs. Using 7,138 samples from the GLODAP v2.2023 dataset, we classify North Atlantic Deep Water, Antarctic Bottom Water, Antarctic Intermediate Water, and Circumpolar Deep Water. Leave-One-Cruise-Out cross-validation confirms spatial generalization (median 73.0%). The 91‰ Δ¹⁴C difference between recently-ventilated and aged water masses encodes approximately 870 years of circulation history. This temporal information, unavailable from physical properties, enables isotope-based classification. Our results suggest that isotopic fingerprints encode sufficient information for water mass identification, with potential implications for paleoceanographic reconstructions where physical properties are unavailable.

![Isotope-only classification of ocean water masses](figures/fig_1_isotope_classification.png)

**Figure 1.** Isotope-only classification of ocean water masses. (a) Δ¹⁴C distributions by water mass showing distinct signatures for "young" (AAIW, NADW) versus "old" (CDW, AABW) water masses. (b) Isotope fingerprint space (δ¹³C vs Δ¹⁴C) revealing water mass clustering. (c) Confusion matrix for isotope-only classification (74.1% accuracy). (d) Accuracy comparison across feature sets.

---

**Full paper coming soon.** Contact christiangruber1234@gmail.com if you're interested in early access or collaboration.

---

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
