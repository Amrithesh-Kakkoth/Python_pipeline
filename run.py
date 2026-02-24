#!/usr/bin/env python3
"""CLI entry point for the pet re-identification pipeline.

Usage:
    # Batch mode — cluster all photos from scratch
    python run.py batch --input /path/to/photos --species dog --output /path/to/clusters --gpu

    # Incremental mode — add new photos to existing clusters
    python run.py add --input /path/to/new_photos --state /path/to/clusters/state.npz --output /path/to/clusters --gpu
"""

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from pet_pipeline.config import SpeciesConfig, SPECIES_DEFAULTS, IMAGE_EXTENSIONS
from pet_pipeline.detection import BodyDetector
from pet_pipeline.pipeline import PetPipeline
from pet_pipeline.state import StateManager


def build_config(args) -> SpeciesConfig:
    """Build SpeciesConfig from CLI args, starting from species defaults."""
    config = SpeciesConfig(**asdict(SPECIES_DEFAULTS[args.species]))

    # Override model paths if provided
    if hasattr(args, 'face_detect_model') and args.face_detect_model:
        config.face_detect_model = args.face_detect_model
    if hasattr(args, 'face_embed_model') and args.face_embed_model:
        config.face_embed_model = args.face_embed_model
    if hasattr(args, 'body_embed_model') and args.body_embed_model:
        config.body_embed_model = args.body_embed_model
    if hasattr(args, 'body_detect_model') and args.body_detect_model:
        config.body_detect_model = args.body_detect_model

    return config


def auto_detect_species(input_dir: Path, gpu: bool = False) -> str:
    """Auto-detect species by running body detector on a sample."""
    default_body_model = str(Path.home() / "pet_embedding" / "yolov8n.pt")
    detector = BodyDetector(model_path=default_body_model, coco_class=16)

    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))

    if not image_paths:
        print("No images found for species detection, defaulting to 'dog'")
        return "dog"

    species = detector.detect_batch_species(image_paths, sample_size=50)
    print(f"Auto-detected species: {species}")
    return species


def main():
    parser = argparse.ArgumentParser(
        description="Production pet re-identification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run.py batch --input /photos --species dog --output /clusters --gpu
  python run.py add --input /new_photos --state /clusters/state.npz --output /clusters
""")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command")

    # Batch subcommand
    batch_p = sub.add_parser("batch", help="Cluster all photos from scratch")
    batch_p.add_argument("--input", "-i", required=True, help="Input image directory")
    batch_p.add_argument("--output", "-o", required=True, help="Output cluster directory")
    batch_p.add_argument("--species", "-s", choices=["dog", "cat"], default=None,
                         help="Species (auto-detected if not given)")
    batch_p.add_argument("--gpu", action="store_true", help="Use GPU for ONNX inference")
    batch_p.add_argument("--face-detect-model", default=None, help="Custom face detection model")
    batch_p.add_argument("--face-embed-model", default=None, help="Custom face embedding model")
    batch_p.add_argument("--body-embed-model", default=None, help="Custom body embedding model")
    batch_p.add_argument("--body-detect-model", default=None, help="Custom body detection model")

    # Incremental subcommand
    add_p = sub.add_parser("add", help="Add new photos to existing clusters")
    add_p.add_argument("--input", "-i", required=True, help="Input directory with new images")
    add_p.add_argument("--state", required=True, help="Path to existing state.npz")
    add_p.add_argument("--output", "-o", required=True, help="Output cluster directory")
    add_p.add_argument("--gpu", action="store_true", help="Use GPU for ONNX inference")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "batch":
        input_dir = Path(args.input)
        output_dir = Path(args.output)

        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            sys.exit(1)

        # Auto-detect species if needed
        species = args.species
        if species is None:
            species = auto_detect_species(input_dir, args.gpu)
        args.species = species

        config = build_config(args)
        pipeline = PetPipeline(config, gpu=args.gpu)
        pipeline.run_batch(input_dir, output_dir)

    elif args.command == "add":
        input_dir = Path(args.input)
        state_path = args.state
        output_dir = Path(args.output)

        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            sys.exit(1)
        if not Path(state_path).exists():
            print(f"ERROR: State file does not exist: {state_path}")
            sys.exit(1)

        # Load config from state
        state = StateManager.load(state_path)
        config = state.config
        pipeline = PetPipeline(config, gpu=args.gpu)
        pipeline.run_incremental(input_dir, state_path, output_dir)


if __name__ == "__main__":
    main()
