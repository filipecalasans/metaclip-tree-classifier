"""
Example: classify a plant image and search a folder by text query.
Run from the project root:
    python examples/classify_plant.py
"""

from plant_classifier import PlantClassifier
from plant_classifier.search import PlantSearchIndex


def example_classify():
    classifier = PlantClassifier(variant="base")

    # Replace with a real image path
    image_path = "data/sample.jpg"
    results = classifier.classify(image_path, top_k=5)

    print("Classification results:")
    for species, prob in results:
        print(f"  {species:<35} {prob:.1%}")


def example_search():
    index = PlantSearchIndex(variant="base")
    index.add_directory("data/images/")

    query = "flour"
    results = index.search(query, top_k=3)

    print(f'\nSearch results for "{query}":')
    for path, score in results:
        print(f"  [{score:.3f}]  {path}")


if __name__ == "__main__":
    example_classify()
    example_search()
