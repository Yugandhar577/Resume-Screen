import praw
import re
from typing import List, Tuple, Optional

class RedditResumeScraper:
    """Scrape r/resumes for before/after resume improvement pairs."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def extract_resume_pairs(self, limit: int = 1000) -> List[Tuple[str, str]]:
        """
        Scrape r/resumes for original + improved resume bullet pairs.
        Returns: [(original_bullet, improved_bullet)]
        """
        pairs = []
        subreddit = self.reddit.subreddit("resumes")
        
        # Look for posts with "roast", "feedback", "critique" in title
        keywords = ["roast", "feedback", "critique", "help", "review"]
        
        for submission in subreddit.search("flair:feedback", limit=limit):
            if any(keyword in submission.title.lower() for keyword in keywords):
                # Look for top comments with improvements
                submission.comments.replace_more(limit=0)
                
                for comment in submission.comments[:5]:  # Top 5 comments
                    if self._is_improvement_comment(comment.body):
                        original_bullets = self._extract_bullets_from_post(submission.selftext)
                        improved_bullets = self._extract_bullets_from_comment(comment.body)
                        
                        # Pair up bullets
                        for orig, improved in zip(original_bullets, improved_bullets):
                            if len(orig) > 20 and len(improved) > 20:  # Filter short bullets
                                pairs.append((orig, improved))
        
        return pairs
    
    def _is_improvement_comment(self, comment_text: str) -> bool:
        """Check if comment contains resume improvements."""
        improvement_indicators = [
            "change to:", "rewrite as:", "better:", "improved:",
            "instead of:", "try:", "→", "->", "should be:"
        ]
        return any(indicator in comment_text.lower() for indicator in improvement_indicators)
    
    def _extract_bullets_from_post(self, post_text: str) -> List[str]:
        """Extract bullet points from original post."""
        # Look for lines starting with • - * or numbered
        bullet_pattern = r'^[\s]*[•\-\*\d\.]+\s*(.+)$'
        bullets = []
        
        for line in post_text.split('\n'):
            match = re.match(bullet_pattern, line.strip())
            if match:
                bullets.append(match.group(1).strip())
        
        return bullets
    
    def _extract_bullets_from_comment(self, comment_text: str) -> List[str]:
        """Extract improved bullets from comment."""
        # Similar pattern but look for quoted or formatted improvements
        bullets = []
        lines = comment_text.split('\n')
        
        for line in lines:
            # Look for quoted improvements or bullets
            if line.strip().startswith('"') or line.strip().startswith('•'):
                clean_line = line.strip().strip('"').strip('•').strip()
                if len(clean_line) > 20:
                    bullets.append(clean_line)
        
        return bullets

# Usage example (requires Reddit API credentials)
def scrape_resume_pairs():
    """
    To use this, you need to:
    1. Create Reddit app at https://www.reddit.com/prefs/apps
    2. Get client_id and client_secret
    3. Set user_agent to something descriptive
    """
    # scraper = RedditResumeScraper(
    #     client_id="your_client_id",
    #     client_secret="your_client_secret", 
    #     user_agent="ResumeML/1.0"
    # )
    # pairs = scraper.extract_resume_pairs(limit=500)
    # return pairs
    pass