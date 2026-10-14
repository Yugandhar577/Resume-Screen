"""Public Arbeitnow API, with source attribution, bounded fetch, and shared caching."""
import json
import re
import time
from html import unescape
from html.parser import HTMLParser
from threading import Lock
from urllib.parse import urlparse
import requests


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.hidden += 1
        if tag in ('p', 'li', 'br', 'div', 'h1', 'h2', 'h3'): self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.hidden = max(0, self.hidden - 1)
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)


def plain_text(content):
    parser = PlainText(); parser.feed(content)
    return re.sub(r'\n\s*\n+', '\n\n', unescape(''.join(parser.parts))).strip()


class Arbeitnow:
    endpoint = 'https://www.arbeitnow.com/api/job-board-api'
    def __init__(self):
        self.cached = None; self.updated = 0; self.lock = Lock()

    def search(self, query='', limit=10):
        with self.lock:
            if self.cached is None or time.monotonic() - self.updated > 21600:
                with requests.get(self.endpoint, timeout=15, stream=True, allow_redirects=False) as response:
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_content(65536):
                        body.extend(chunk)
                        if len(body) > 4 * 1024 * 1024:
                            raise ValueError('Job feed exceeds the allowed size.')
                    data = json.loads(body)
                jobs = []
                for item in data.get('data', [])[:100]:
                    url = item.get('url', '')
                    parsed = urlparse(url)
                    if parsed.scheme != 'https' or parsed.hostname not in ('arbeitnow.com', 'www.arbeitnow.com'):
                        continue
                    jobs.append({'id': item['slug'], 'title': item['title'], 'company': item['company_name'],
                                 'description': plain_text(item.get('description', ''))[:50000],
                                 'url': url, 'location': item.get('location', ''), 'remote': bool(item.get('remote')),
                                 'source': 'Arbeitnow', 'source_url': 'https://www.arbeitnow.com'})
                self.cached, self.updated = jobs, time.monotonic()
        words = query.casefold().split()
        selected = [job for job in self.cached if all(word in (job['title']+' '+job['description']).casefold() for word in words)]
        return {'jobs': selected[:limit], 'source': 'Arbeitnow', 'source_url': 'https://www.arbeitnow.com',
                'note': 'Public API listings, cached for six hours. Check location, language, and eligibility on the source listing. No applications are submitted.'}
