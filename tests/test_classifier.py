import pytest
import torch
from unittest.mock import MagicMock, patch
from PIL import Image

from plant_classifier.classifier import PlantClassifier
from plant_classifier.species import PLANT_SPECIES


@pytest.fixture
def mock_encoder():
    with patch("plant_classifier.classifier.MetaCLIPEncoder") as MockEncoder:
        instance = MockEncoder.return_value
        dim = 512
        n = len(PLANT_SPECIES)
        # encode_texts returns normalized random vectors
        instance.encode_texts.return_value = torch.randn(n, dim)
        instance.encode_image.return_value = torch.randn(1, dim)
        yield instance


def test_classify_returns_top_k(mock_encoder):
    classifier = PlantClassifier()
    classifier._text_features = torch.nn.functional.normalize(
        torch.randn(len(PLANT_SPECIES), 512), dim=-1
    )
    image = MagicMock(spec=Image.Image)
    mock_encoder.encode_image.return_value = torch.nn.functional.normalize(
        torch.randn(1, 512), dim=-1
    )
    classifier.encoder = mock_encoder

    results = classifier.classify(image, top_k=3)

    assert len(results) == 3
    for species, prob in results:
        assert species in PLANT_SPECIES
        assert 0.0 <= prob <= 1.0


def test_classify_probabilities_sum_to_one(mock_encoder):
    classifier = PlantClassifier()
    classifier._text_features = torch.nn.functional.normalize(
        torch.randn(len(PLANT_SPECIES), 512), dim=-1
    )
    image = MagicMock(spec=Image.Image)
    mock_encoder.encode_image.return_value = torch.nn.functional.normalize(
        torch.randn(1, 512), dim=-1
    )
    classifier.encoder = mock_encoder

    results = classifier.classify(image, top_k=len(PLANT_SPECIES))
    total = sum(prob for _, prob in results)

    assert abs(total - 1.0) < 1e-4


def test_classify_sorted_by_confidence(mock_encoder):
    classifier = PlantClassifier()
    classifier._text_features = torch.nn.functional.normalize(
        torch.randn(len(PLANT_SPECIES), 512), dim=-1
    )
    mock_encoder.encode_image.return_value = torch.nn.functional.normalize(
        torch.randn(1, 512), dim=-1
    )
    classifier.encoder = mock_encoder

    results = classifier.classify(MagicMock(spec=Image.Image), top_k=5)
    probs = [prob for _, prob in results]

    assert probs == sorted(probs, reverse=True)
