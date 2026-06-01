"""
Pairwise cosine similarity analysis for a species list against MetaCLIP text embeddings.

Run from the project root:
    python scripts/pairwise_similarity.py
"""

import torch
from plant_classifier.encoder import MetaCLIPEncoder

PNW_TREES = [
    # Conifers
    "Douglas fir",
    "western red cedar",
    "Sitka spruce",
    "western hemlock",
    "grand fir",
    "subalpine fir",
    "Pacific silver fir",
    "lodgepole pine",
    "western white pine",
    "ponderosa pine",
    "sugar pine",
    "whitebark pine",
    "mountain hemlock",
    "Alaska yellow cedar",
    "Port Orford cedar",
    "Pacific yew",
    "western larch",
    # Broadleaf / deciduous
    "bigleaf maple",
    "vine maple",
    "red alder",
    "black cottonwood",
    "quaking aspen",
    "Pacific madrone",
    "Oregon white oak",
    "bitter cherry",
    "cascara",
    "Pacific dogwood",
    "Oregon ash",
    "black hawthorn",
]

PROMPT_TEMPLATE = "a photo of a {} tree"


def compute_similarity(encoder: MetaCLIPEncoder, species: list[str]) -> torch.Tensor:
    prompts = [PROMPT_TEMPLATE.format(s) for s in species]
    feats = encoder.encode_texts(prompts)  # (N, D), already L2-normalised
    return feats @ feats.T                 # (N, N) cosine similarity


def print_matrix(sim: torch.Tensor, species: list[str]) -> None:
    col_w = 22
    header = " " * col_w + "".join(f"{s[:10]:>12}" for s in species)
    print(header)
    for i, label in enumerate(species):
        row = f"{label:<{col_w}}" + "".join(f"{sim[i, j].item():12.3f}" for j in range(len(species)))
        print(row)


def print_top_confusions(sim: torch.Tensor, species: list[str], threshold: float = 0.90) -> None:
    n = len(species)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = sim[i, j].item()
            if score >= threshold:
                pairs.append((score, species[i], species[j]))

    pairs.sort(reverse=True)

    if not pairs:
        print(f"No pairs above similarity threshold {threshold:.2f}.")
        return

    print(f"{'Score':>7}  {'Species A':<28}  Species B")
    print("-" * 65)
    for score, a, b in pairs:
        print(f"{score:7.3f}  {a:<28}  {b}")


def print_weakest_species(sim: torch.Tensor, species: list[str], bottom_n: int = 5) -> None:
    n = len(species)
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_sim = sim[mask].reshape(n, n - 1).mean(dim=1)

    ranked = sorted(zip(mean_sim.tolist(), species))
    print(f"{'Mean sim':>10}  Species")
    print("-" * 40)
    for score, label in ranked[:bottom_n]:
        print(f"{score:10.3f}  {label}")


def main() -> None:
    print(f"Loading MetaCLIP encoder...")
    encoder = MetaCLIPEncoder(variant="base")

    print(f"Encoding {len(PNW_TREES)} Pacific Northwest tree species...\n")
    sim = compute_similarity(encoder, PNW_TREES)

    print("=" * 65)
    print("PAIRWISE COSINE SIMILARITY MATRIX")
    print("=" * 65)
    print_matrix(sim, PNW_TREES)

    print("\n" + "=" * 65)
    print("HIGH-CONFUSION PAIRS  (similarity >= 0.90)")
    print("=" * 65)
    print_top_confusions(sim, PNW_TREES, threshold=0.90)

    print("\n" + "=" * 65)
    print("WEAKEST REPRESENTATIONS  (lowest mean similarity to all others)")
    print("=" * 65)
    print_weakest_species(sim, PNW_TREES, bottom_n=5)


if __name__ == "__main__":
    main()
