import torch
from pathlib import Path
from PIL import Image

from .encoder import MetaCLIPEncoder


class PlantSearchIndex:
    """
    Builds a semantic search index over a collection of plant images,
    queryable by natural language (e.g. "purple flower with round petals").
    """

    def __init__(self, variant: str = "base"):
        self.encoder = MetaCLIPEncoder(variant=variant)
        self._embeddings: list[torch.Tensor] = []
        self._paths: list[Path] = []

    def add(self, image_path: str | Path):
        path = Path(image_path)
        embedding = self.encoder.encode_image(path).cpu()
        self._embeddings.append(embedding)
        self._paths.append(path)

    def add_directory(self, directory: str | Path, extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")):
        for path in sorted(Path(directory).rglob("*")):
            if path.suffix.lower() in extensions:
                self.add(path)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Path, float]]:
        if not self._embeddings:
            raise RuntimeError("Index is empty — add images before searching.")
        query_features = self.encoder.encode_texts([query]).cpu()
        all_embeddings = torch.cat(self._embeddings, dim=0)  # (N, D)
        scores = (all_embeddings @ query_features.T).squeeze()
        top_indices = scores.argsort(descending=True)[:top_k]
        return [(self._paths[i], scores[i].item()) for i in top_indices]

    def __len__(self) -> int:
        return len(self._paths)
