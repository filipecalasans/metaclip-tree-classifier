import torch
import open_clip
from PIL import Image
from pathlib import Path


class MetaCLIPEncoder:
    """
    Wraps MetaCLIP image and text encoders.
    Automatically selects CUDA/ROCm or CPU.
    """

    MODELS = {
        "base":  ("ViT-B-32-quickgelu", "metaclip_400m"),
        "large": ("ViT-L-14-quickgelu", "metaclip_400m"),
        "huge":  ("ViT-H-14-quickgelu", "metaclip_fullcc"),
    }

    def __init__(self, variant: str = "base"):
        if variant not in self.MODELS:
            raise ValueError(f"variant must be one of {list(self.MODELS)}")

        arch, pretrained = self.MODELS[variant]
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.model.eval().to(self.device)

    def encode_image(self, image: Image.Image | str | Path) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device_type):
            features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.float()

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device_type):
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.float()
