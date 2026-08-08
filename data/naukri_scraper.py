from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from typing import List, Dict

class NaukriScraper:
    """Scrape job descriptions from Naukri.com for Indian market data."""
    
    def __init__(self, headless: bool = True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def scrape_job_listings(self, search_terms: List[str], 
                           max_jobs_per_term: int = 100) -> List[Dict]:
        """
        Scrape job descriptions from Naukri.com.
        Returns list of job dictionaries with title, company, description, skills.
        """
        all_jobs = []
        
        for term in search_terms:
            print(f"Scraping jobs for: {term}")
            jobs = self._scrape_search_term(term, max_jobs_per_term)
            all_jobs.extend(jobs)
            time.sleep(2)  # Be respectful to the server
        
        return all_jobs
    
    def _scrape_search_term(self, search_term: str, max_jobs: int) -> List[Dict]:
        """Scrape jobs for a specific search term."""
        jobs = []
        
        # Navigate to Naukri search
        search_url = f"https://www.naukri.com/{search_term}-jobs"
        self.driver.get(search_url)
        time.sleep(3)
        
        page = 1
        while len(jobs) < max_jobs and page <= 5:  # Limit to 5 pages
            try:
                # Find job cards
                job_cards = self.driver.find_elements(By.CLASS_NAME, "srp-jobtuple-wrapper")
                
                for card in job_cards[:20]:  # Process up to 20 jobs per page
                    try:
                        job_data = self._extract_job_data(card)
                        if job_data:
                            jobs.append(job_data)
                            
                        if len(jobs) >= max_jobs:
                            break
                            
                    except Exception as e:
                        print(f"Error extracting job data: {e}")
                        continue
                
                # Go to next page
                try:
                    next_button = self.driver.find_element(By.CLASS_NAME, "fright")
                    if "Next" in next_button.text:
                        next_button.click()
                        time.sleep(3)
                        page += 1
                    else:
                        break
                except:
                    break
                    
            except Exception as e:
                print(f"Error on page {page}: {e}")
                break
        
        return jobs
    
    def _extract_job_data(self, job_card) -> Dict:
        """Extract job information from a job card element."""
        try:
            # Job title
            title_elem = job_card.find_element(By.CLASS_NAME, "title")
            title = title_elem.text.strip()
            
            # Company name
            company_elem = job_card.find_element(By.CLASS_NAME, "subTitle")
            company = company_elem.text.strip()
            
            # Experience
            try:
                exp_elem = job_card.find_element(By.CLASS_NAME, "expwdth")
                experience = exp_elem.text.strip()
            except:
                experience = ""
            
            # Skills (if available)
            skills = []
            try:
                skill_elements = job_card.find_elements(By.CLASS_NAME, "skill-tag")
                skills = [skill.text.strip() for skill in skill_elements]
            except:
                pass
            
            # Click to get full description
            try:
                title_elem.click()
                time.sleep(2)
                
                # Get job description from detail page
                desc_elem = self.driver.find_element(By.CLASS_NAME, "dang-inner-html")
                description = desc_elem.text.strip()
                
                # Go back
                self.driver.back()
                time.sleep(2)
                
            except:
                description = ""
            
            return {
                "title": title,
                "company": company,
                "experience": experience,
                "skills": skills,
                "description": description,
                "source": "naukri"
            }
            
        except Exception as e:
            print(f"Error extracting job data: {e}")
            return None
    
    def save_jobs(self, jobs: List[Dict], filename: str):
        """Save scraped jobs to CSV."""
        df = pd.DataFrame(jobs)
        df.to_csv(filename, index=False)
        print(f"Saved {len(jobs)} jobs to {filename}")
    
    def close(self):
        """Close the browser."""
        self.driver.quit()

# Usage example
def scrape_indian_jobs():
    """
    Scrape 5000 job descriptions from Naukri.com.
    Provides local market relevance for your project.
    """
    search_terms = [
        "data-scientist",
        "software-engineer", 
        "web-developer",
        "machine-learning",
        "python-developer",
        "java-developer",
        "devops-engineer",
        "frontend-developer",
        "backend-developer",
        "full-stack-developer"
    ]
    
    scraper = NaukriScraper(headless=True)
    
    try:
        jobs = scraper.scrape_job_listings(search_terms, max_jobs_per_term=50)
        scraper.save_jobs(jobs, "naukri_jobs.csv")
        print(f"Total jobs scraped: {len(jobs)}")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    # Uncomment to run scraper
    # scrape_indian_jobs()
    pass