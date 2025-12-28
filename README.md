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

For GPU support:
```bash
pip install -r requirements_gpu.txt
```

## Data

Download the GLODAP v2.2023 Merged Master File from https://www.glodap.info/ and place it in the `data/` directory. See [data/README.md](data/README.md) for details.

## Usage

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
├── src/
│   ├── isotope_classification.py   # Core experiment: Δ¹⁴C + δ¹³C only
│   ├── waterprint_analysis.py      # Full-feature classification
│   └── generate_figures.py         # Reproduce manuscript figures
├── data/
│   └── README.md                   # Data download instructions
├── requirements.txt
├── requirements_gpu.txt
├── LICENSE
└── CITATION.cff
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite using the information in [CITATION.cff](CITATION.cff).
