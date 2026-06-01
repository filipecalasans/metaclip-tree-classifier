# MetaCLIP Plant Classifier

> **Disclaimer:** This project was built as a learning tool and exploratory exercise. It was developed collaboratively with [Claude Code](https://claude.ai/code) (Anthropic's AI coding assistant) as a hands-on way to understand zero-shot image classification with vision-language models. The code, experiments, and analysis were produced interactively through conversation — not as a production system.

---

## Overview

Zero-shot plant species classifier powered by [MetaCLIP](https://github.com/facebookresearch/MetaCLIP). Given an image, the classifier ranks a curated list of species by visual similarity using the model's joint image-text embedding space — no fine-tuning or labeled training data required.

The project includes:
- A reusable `PlantClassifier` and `PlantSearchIndex` library
- A CLI tool (`plant-classify`) for classifying images or searching image folders by text query
- A curated Pacific Northwest tree species list with habitat-aware prompts
- Embedding analysis scripts and reports documenting prompt engineering experiments

---

## Setup

```bash
python3 -m venv .venv
source activate.sh
pip install -e ".[dev]"
```

---

## Usage

### Classify images

```bash
# Default: PNW tree species with habitat-aware prompts
plant-classify classify data/sample.jpg

# Multiple images, top-3 results
plant-classify classify img1.jpg img2.jpg --top-k 3

# General plant species list
plant-classify --species-set general classify data/sample.jpg

# Larger model
plant-classify --variant large classify data/sample.jpg
```

### Search a folder by text query

```bash
plant-classify search data/images/ "subalpine fir at treeline"
plant-classify search data/images/ "coastal rainforest conifer" --top-k 3
```

### Model variants

| Flag | Architecture | Pretrained on |
|---|---|---|
| `base` (default) | ViT-B-32 | MetaCLIP 400M |
| `large` | ViT-L-14 | MetaCLIP 400M |
| `huge` | ViT-H-14 | MetaCLIP FullCC |

---

## Species lists

Two species sets are available via `--species-set`:

| Set | Flag | Description |
|---|---|---|
| PNW trees | `pnw` (default) | 28 Pacific Northwest tree species with habitat-aware prompts |
| General plants | `general` | ~60 common plant species using generic prompt templates |

The PNW list covers 17 conifers and 11 broadleaf/deciduous species commonly found in the Pacific Northwest. See [`src/plant_classifier/species.py`](src/plant_classifier/species.py) for the full list and prompts.

---

## Experiments

### How it works

MetaCLIP is a vision-language model trained on hundreds of millions of image-text pairs. Classification is zero-shot: the model encodes both the query image and a set of text descriptions, then ranks species by cosine similarity in the shared embedding space. No images of the target species are needed at inference time.

The text prompt design directly affects accuracy. Two strategies were evaluated:

---

### Experiment 1 — Generic prompt template

All species encoded with a shared template: `"a photo of a {} tree"`.

Key finding: **18 high-confusion pairs** (cosine similarity ≥ 0.90) were identified. The worst was western hemlock / mountain hemlock at **0.969** — nearly indistinguishable. Large clusters of firs (0.926–0.940) and pines (0.900–0.916) also showed high inter-species confusion.

Full results: [`scripts/pnw_trees_analysis.md`](scripts/pnw_trees_analysis.md)
Reproduction: `python scripts/pairwise_similarity.py`

---

### Experiment 2 — Habitat-aware prompts

Each species was assigned a unique prompt embedding its ecological habitat, elevation, and geographic range (e.g. `"a photo of a mountain hemlock tree at a subalpine treeline with heavy snow"`).

Key finding: High-confusion pairs dropped from **18 to 2** — an 89% reduction. The hemlock pair fell from 0.969 to 0.782. The two remaining pairs (Douglas fir / grand fir at 0.912, grand fir / Pacific silver fir at 0.904) share ecologically adjacent habitats and likely require visual morphology descriptors to resolve further.

Full results: [`scripts/pnw_trees_habitat_analysis.md`](scripts/pnw_trees_habitat_analysis.md)
Reproduction: `python scripts/pairwise_similarity_habitat.py`

---

### Key takeaway

Habitat context in text prompts is a highly effective zero-shot specialization strategy for conifer species that occupy distinct ecological niches. For species defined primarily by visual morphology (bark texture, cone shape) or that co-occur in the same habitat, text prompts alone may be insufficient — supplementing with example images via `PlantSearchIndex` is recommended.

---

## Project structure

```
├── src/plant_classifier/
│   ├── encoder.py       # MetaCLIP image and text encoder wrapper
│   ├── classifier.py    # Zero-shot classifier with prompt ensembling
│   ├── search.py        # Text-to-image search index
│   ├── species.py       # PNW_TREES and PLANT_SPECIES definitions
│   └── cli.py           # plant-classify entry point
├── scripts/
│   ├── pairwise_similarity.py           # Baseline analysis (generic prompts)
│   ├── pairwise_similarity_habitat.py   # Habitat-aware prompt analysis
│   ├── pnw_trees_analysis.md            # Baseline results report
│   └── pnw_trees_habitat_analysis.md    # Habitat-aware results report
├── examples/
│   └── classify_plant.py    # Usage examples
└── tests/
    └── test_classifier.py
```

---

## Dependencies

- [open_clip](https://github.com/mlfoundations/open_clip) — MetaCLIP model loading and inference
- PyTorch — tensor operations and GPU support
- Pillow — image loading and preprocessing
