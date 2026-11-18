"""Train only from explicit reviewed pairs; never save an untrained checkpoint."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/t5_training_data.csv")
    parser.add_argument("--output", default="ml/models/t5_rewriter")
    parser.add_argument('--provenance', required=True, help='Documented source/license and factual review manifest')
    parser.add_argument('--allow-ai-reviewed-synthetic', action='store_true', help='Explicit experimental mode; never constitutes human review')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        raise ValueError('Use 1–100 training epochs.')
    from training.validate_reviewed_data import validate_manifest
    provenance = validate_manifest(args.provenance, 'resume_rewriting', args.data, args.allow_ai_reviewed_synthetic)
    from ml.rewriter.t5_rewriter import T5ResumeRewriter, create_training_data_from_sources
    pairs = create_training_data_from_sources(args.data)
    import torch
    torch.set_num_threads(4)
    T5ResumeRewriter().train(pairs, args.output, provenance=provenance, allow_ai_reviewed_synthetic=args.allow_ai_reviewed_synthetic, epochs=args.epochs)


if __name__ == "__main__":
    main()
