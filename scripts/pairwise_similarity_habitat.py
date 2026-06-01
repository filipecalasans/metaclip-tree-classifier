"""
Pairwise cosine similarity analysis using per-species habitat-aware prompts.
Companion to pairwise_similarity.py — uses specific ecological context per
species instead of a generic template, to better differentiate confused pairs.

Run from the project root:
    python scripts/pairwise_similarity_habitat.py
"""

import torch
from plant_classifier.encoder import MetaCLIPEncoder

# Maps display name -> habitat-aware prompt
PNW_TREES: dict[str, str] = {
    # Conifers
    "Douglas fir":        "a photo of a Douglas fir tree in a Pacific Northwest lowland mixed forest",
    "western red cedar":  "a photo of a western red cedar tree in a coastal rainforest with hanging moss",
    "Sitka spruce":       "a photo of a Sitka spruce tree in a coastal fog belt rainforest",
    "western hemlock":    "a photo of a western hemlock tree in a shaded coastal rainforest understory",
    "grand fir":          "a photo of a grand fir tree in a low-elevation moist mixed forest",
    "subalpine fir":      "a photo of a subalpine fir tree at high-elevation treeline",
    "Pacific silver fir": "a photo of a Pacific silver fir tree in a mid-elevation Cascade montane forest",
    "lodgepole pine":     "a photo of a lodgepole pine tree in a dry subalpine or post-fire forest",
    "western white pine": "a photo of a western white pine tree in a mixed conifer forest",
    "ponderosa pine":     "a photo of a ponderosa pine tree in an open dry east-side forest",
    "sugar pine":         "a photo of a sugar pine tree in a southern Cascade mixed conifer forest with large cones",
    "whitebark pine":     "a photo of a whitebark pine tree at an alpine treeline",
    "mountain hemlock":   "a photo of a mountain hemlock tree at a subalpine treeline with heavy snow",
    "Alaska yellow cedar":"a photo of an Alaska yellow cedar tree in a coastal subalpine bog",
    "Port Orford cedar":  "a photo of a Port Orford cedar tree in a southwest Oregon riparian forest",
    "Pacific yew":        "a photo of a Pacific yew tree as a shaded forest understory conifer",
    "western larch":      "a photo of a western larch deciduous conifer tree in an east Cascade dry forest",
    # Broadleaf / deciduous
    "bigleaf maple":      "a photo of a bigleaf maple tree with large leaves in a Pacific Northwest riparian forest",
    "vine maple":         "a photo of a vine maple small tree in a shaded forest understory",
    "red alder":          "a photo of a red alder tree along a Pacific Northwest stream or riparian area",
    "black cottonwood":   "a photo of a black cottonwood tree along a river floodplain",
    "quaking aspen":      "a photo of a quaking aspen tree in a clonal grove on the east side of the Cascades",
    "Pacific madrone":    "a photo of a Pacific madrone evergreen tree with red peeling bark on a coastal bluff",
    "Oregon white oak":   "a photo of an Oregon white oak tree in a dry savanna or woodland",
    "bitter cherry":      "a photo of a bitter cherry small tree in a disturbed or forest edge habitat",
    "cascara":            "a photo of a cascara buckthorn small tree in a Pacific Northwest riparian woodland",
    "Pacific dogwood":    "a photo of a Pacific dogwood flowering tree in a shaded forest understory",
    "Oregon ash":         "a photo of an Oregon ash tree in a wet lowland forest or floodplain",
    "black hawthorn":     "a photo of a black hawthorn thorny shrub tree along a Pacific Northwest stream",
}


def compute_similarity(encoder: MetaCLIPEncoder, species: dict[str, str]) -> torch.Tensor:
    prompts = list(species.values())
    feats = encoder.encode_texts(prompts)  # (N, D), already L2-normalised
    return feats @ feats.T                 # (N, N) cosine similarity


def print_matrix(sim: torch.Tensor, labels: list[str]) -> None:
    col_w = 22
    header = " " * col_w + "".join(f"{s[:10]:>12}" for s in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:<{col_w}}" + "".join(f"{sim[i, j].item():12.3f}" for j in range(len(labels)))
        print(row)


def print_top_confusions(sim: torch.Tensor, labels: list[str], threshold: float = 0.90) -> None:
    n = len(labels)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = sim[i, j].item()
            if score >= threshold:
                pairs.append((score, labels[i], labels[j]))

    pairs.sort(reverse=True)

    if not pairs:
        print(f"No pairs above similarity threshold {threshold:.2f}.")
        return

    print(f"{'Score':>7}  {'Species A':<28}  Species B")
    print("-" * 65)
    for score, a, b in pairs:
        print(f"{score:7.3f}  {a:<28}  {b}")


def print_weakest_species(sim: torch.Tensor, labels: list[str], bottom_n: int = 5) -> None:
    n = len(labels)
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_sim = sim[mask].reshape(n, n - 1).mean(dim=1)

    ranked = sorted(zip(mean_sim.tolist(), labels))
    print(f"{'Mean sim':>10}  Species")
    print("-" * 40)
    for score, label in ranked[:bottom_n]:
        print(f"{score:10.3f}  {label}")


def main() -> None:
    print("Loading MetaCLIP encoder...")
    encoder = MetaCLIPEncoder(variant="base")

    labels = list(PNW_TREES.keys())
    print(f"Encoding {len(labels)} Pacific Northwest tree species with habitat-aware prompts...\n")
    sim = compute_similarity(encoder, PNW_TREES)

    print("=" * 65)
    print("PAIRWISE COSINE SIMILARITY MATRIX")
    print("=" * 65)
    print_matrix(sim, labels)

    print("\n" + "=" * 65)
    print("HIGH-CONFUSION PAIRS  (similarity >= 0.90)")
    print("=" * 65)
    print_top_confusions(sim, labels, threshold=0.90)

    print("\n" + "=" * 65)
    print("WEAKEST REPRESENTATIONS  (lowest mean similarity to all others)")
    print("=" * 65)
    print_weakest_species(sim, labels, bottom_n=5)


if __name__ == "__main__":
    main()
