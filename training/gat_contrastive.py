"""Compatibility entry point for the single maintained GAT implementation."""
from ml.gat.model import ResumeGAT, ContrastiveLoss
from ml.gat.train import GATTrainer, main

if __name__ == "__main__":
    main()
