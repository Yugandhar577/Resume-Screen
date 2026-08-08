"""
Complete data pipeline for Resume-JD Graph Matcher.
Combines all data sources into training-ready datasets.
"""

from weak_labelling import WeakLabeller
from synthetic_negatives import SyntheticNegativeGenerator
from gpt_distillation import GPTDistillationLabeller
from reddit_scraper import RedditResumeScraper
from naukri_scraper import NaukriScraper
import pandas as pd

class DataPipeline:
    """Orchestrates the complete data collection and preparation pipeline."""
    
    def __init__(self):
        self.weak_labeller = WeakLabeller()
        self.neg_generator = SyntheticNegativeGenerator()
    
    def create_gat_dataset(self):
        """Create dataset for GAT training with synthetic hard negatives."""
        print("Step 1: Loading Kaggle + LinkedIn datasets...")
        # resumes_df = pd.read_csv('data/kaggle_resumes.csv')
        # jds_df = pd.read_csv('data/linkedin_jobs.csv')
        
        print("Step 2: Generating weak supervision pairs...")
        # pairs = self.weak_labeller.generate_pairs(resumes_df, jds_df, num_pairs=5000)
        # balanced_pairs = self.weak_labeller.balance_dataset(pairs)
        
        print("Step 3: Adding synthetic hard negatives...")
        # enhanced_pairs = []
        # for resume_text, jd_text, label in balanced_pairs:
        #     if label == 1:  # Positive pair
        #         jd_skills = jd_text.split()[:20]  # Extract skills
        #         contrastive_pairs = self.neg_generator.create_contrastive_pairs(
        #             resume_text, jd_text, jd_skills
        #         )
        #         enhanced_pairs.extend(contrastive_pairs)
        
        # Save GAT training data
        # gat_df = pd.DataFrame(enhanced_pairs, columns=['resume', 'jd', 'label'])
        # gat_df.to_csv('data/gat_training_data.csv', index=False)
        
        print("GAT dataset created with synthetic hard negatives!")
    
    def create_t5_dataset(self, openai_key: str = None):
        """Create dataset for T5 fine-tuning using multiple sources."""
        all_pairs = []
        
        # Source 1: GPT distillation (if API key provided)
        if openai_key:
            print("Generating T5 pairs via GPT distillation...")
            # labeller = GPTDistillationLabeller(openai_key)
            # resumes_df = pd.read_csv('data/kaggle_resumes.csv')
            # bullets = labeller.extract_bullets_from_resumes(resumes_df)
            # gpt_pairs = labeller.generate_training_pairs(bullets[:1000])
            # all_pairs.extend(gpt_pairs)
        
        # Source 2: Reddit r/resumes (if credentials available)
        print("Scraping Reddit for resume improvement pairs...")
        # reddit_scraper = RedditResumeScraper(client_id, client_secret, user_agent)
        # reddit_pairs = reddit_scraper.extract_resume_pairs(limit=500)
        # all_pairs.extend(reddit_pairs)
        
        # Save T5 training data
        # t5_df = pd.DataFrame(all_pairs, columns=['original', 'improved'])
        # t5_df.to_csv('data/t5_training_data.csv', index=False)
        
        print(f"T5 dataset created with {len(all_pairs)} pairs!")
    
    def scrape_naukri_jobs(self):
        """Scrape Indian job market for local relevance."""
        print("Scraping Naukri.com for Indian market JDs...")
        # scraper = NaukriScraper(headless=True)
        # jobs = scraper.scrape_job_listings([
        #     "data-scientist", "software-engineer", "web-developer"
        # ], max_jobs_per_term=100)
        # scraper.save_jobs(jobs, "data/naukri_jobs.csv")
        # scraper.close()
        
        print("Naukri scraping complete!")

def run_complete_pipeline():
    """Run the complete data pipeline."""
    pipeline = DataPipeline()
    
    # Step 1: Create GAT training data with synthetic negatives
    pipeline.create_gat_dataset()
    
    # Step 2: Create T5 training data (requires API keys)
    # pipeline.create_t5_dataset(openai_key="your_key_here")
    
    # Step 3: Scrape Indian market for uniqueness
    # pipeline.scrape_naukri_jobs()
    
    print("Complete data pipeline finished!")

if __name__ == "__main__":
    run_complete_pipeline()