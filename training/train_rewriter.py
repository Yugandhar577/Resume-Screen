"""Train only from explicit reviewed pairs; never save an untrained checkpoint."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/t5_training_data.csv")
    parser.add_argument("--output", default="ml/models/t5_rewriter")
    parser.add_argument('--provenance', required=True, help='Documented source/license and factual review manifest')
    args = parser.parse_args()
    from training.validate_reviewed_data import validate_manifest
    provenance = validate_manifest(args.provenance, 'resume_rewriting', args.data)
    from ml.rewriter.t5_rewriter import T5ResumeRewriter, create_training_data_from_sources
    pairs = create_training_data_from_sources(args.data)
    T5ResumeRewriter().train(pairs, args.output, provenance=provenance)


if __name__ == "__main__":
    main()
