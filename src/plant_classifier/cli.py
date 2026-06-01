import argparse
import sys
from pathlib import Path

from .classifier import PlantClassifier
from .search import PlantSearchIndex
from .species import PLANT_SPECIES, PNW_TREES

SPECIES_SETS = {
    "pnw": PNW_TREES,
    "general": PLANT_SPECIES,
}


def cmd_classify(args):
    print(f"Loading MetaCLIP ({args.variant})...")
    classifier = PlantClassifier(species=SPECIES_SETS[args.species_set], variant=args.variant)

    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            print(f"[skip] {path} — file not found", file=sys.stderr)
            continue
        results = classifier.classify(path, top_k=args.top_k)
        print(f"\n{path.name}")
        for rank, (species, prob) in enumerate(results, 1):
            bar = "#" * int(prob * 30)
            print(f"  {rank}. {species:<30} {prob:6.1%}  {bar}")


def cmd_search(args):
    index = PlantSearchIndex(variant=args.variant)
    print(f"Indexing images in {args.directory} ...")
    index.add_directory(args.directory)
    print(f"Indexed {len(index)} images.")

    results = index.search(args.query, top_k=args.top_k)
    print(f"\nResults for: \"{args.query}\"")
    for rank, (path, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.3f}]  {path}")


def main():
    parser = argparse.ArgumentParser(prog="plant-classify", description="Plant species classifier powered by MetaCLIP")
    parser.add_argument("--variant", choices=["base", "large", "huge"], default="base")
    parser.add_argument("--species-set", choices=["pnw", "general"], default="pnw",
                        help="Species list to classify against (default: pnw)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_classify = sub.add_parser("classify", help="Classify one or more images")
    p_classify.add_argument("images", nargs="+", help="Image file paths")
    p_classify.add_argument("--top-k", type=int, default=5)

    p_search = sub.add_parser("search", help="Search a folder of images by text query")
    p_search.add_argument("directory", help="Directory of plant images")
    p_search.add_argument("query", help="Natural language query")
    p_search.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "classify":
        cmd_classify(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
