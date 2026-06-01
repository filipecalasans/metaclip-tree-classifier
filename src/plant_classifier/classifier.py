import torch
from PIL import Image
from pathlib import Path

from .encoder import MetaCLIPEncoder
from .species import PLANT_SPECIES, PNW_TREES


class PlantClassifier:
    """
    Zero-shot plant species classifier backed by MetaCLIP.
    Species labels are pre-encoded at construction time for fast inference.

    Pass a dict[str, str] (name -> prompt) to use per-species habitat-aware
    prompts directly. Pass a list[str] to use template ensembling instead.
    Defaults to the PNW tree species with habitat prompts.
    """

    PROMPT_TEMPLATES = [
        "a photo of a {}",
        "a close-up photo of a {} plant",
        "a botanical photo of {}",
        "a {} leaf",
    ]

    def __init__(self, species: list[str] | dict[str, str] | None = None, variant: str = "base"):
        self.encoder = MetaCLIPEncoder(variant=variant)
        if isinstance(species, dict):
            self.species = list(species.keys())
            self._prompts: list[str] | None = list(species.values())
        else:
            self.species = species or list(PNW_TREES.keys())
            self._prompts = list(PNW_TREES.values()) if species is None else None
        self._text_features = self._encode_species()

    def _encode_species(self) -> torch.Tensor:
        if self._prompts is not None:
            return self.encoder.encode_texts(self._prompts)
        # Template ensembling for plain species name lists
        all_features = []
        for template in self.PROMPT_TEMPLATES:
            prompts = [template.format(s) for s in self.species]
            features = self.encoder.encode_texts(prompts)  # (N, D)
            all_features.append(features)
        stacked = torch.stack(all_features, dim=0)   # (T, N, D)
        averaged = stacked.mean(dim=0)               # (N, D)
        averaged /= averaged.norm(dim=-1, keepdim=True)
        return averaged

    def classify(
        self,
        image: Image.Image | str | Path,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Returns top_k (species, probability) pairs sorted by confidence.
        """
        image_features = self.encoder.encode_image(image)      # (1, D)
        logits = (image_features @ self._text_features.T) * 100
        probs = logits.softmax(dim=-1).squeeze()

        top_indices = probs.argsort(descending=True)[:top_k]
        return [(self.species[i], probs[i].item()) for i in top_indices]
