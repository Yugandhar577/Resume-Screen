"""Prepare category-labeled pairs from user-supplied CSVs. No scraping on import."""
import argparse
from pathlib import Path
import pandas as pd
from data.weak_labelling import WeakLabeller


class DataPipeline:
    def create_gat_dataset(self, resumes_path, jobs_path, output_path="data/gat_training_data.csv", num_pairs=5000):
        labeller = WeakLabeller()
        resumes, jobs = labeller.load_datasets(resumes_path, jobs_path)
        if not {"Category", "Resume_str"}.issubset(resumes.columns):
            raise ValueError("Resume CSV requires Category and Resume_str columns.")
        if not {"title", "description"}.issubset(jobs.columns):
            raise ValueError("Jobs CSV requires title and description columns.")
        pairs = labeller.generate_pairs(resumes, jobs, num_pairs=num_pairs)
        frame = pd.DataFrame(pairs, columns=["resume", "jd", "label"]).drop_duplicates(["resume", "jd"])
        if frame.empty or frame["label"].nunique() != 2:
            raise ValueError("Need positive and negative weakly labeled pairs. Supply more varied data.")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return frame


def main():
    parser = argparse.ArgumentParser(description="Prepare weakly labeled pairs; these are not hiring-outcome labels.")
    parser.add_argument("--resumes", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", default="data/gat_training_data.csv")
    args = parser.parse_args()
    frame = DataPipeline().create_gat_dataset(args.resumes, args.jobs, args.output)
    print(f"Wrote {len(frame)} weakly labeled pairs to {args.output}. Split before creating synthetic negatives.")


if __name__ == "__main__":
    main()
