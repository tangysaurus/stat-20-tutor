import requests
import re
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    """Remove extra new lines and whitespace"""
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def clean_link(text):
    """Get relative link"""
    text = re.search(r"[^./].*", text).group(0)
    return text

def scrape():
    """Web scrape Stat 20 lecture notes by retrieving content and next page links"""
    rel_url = "https://stat20.berkeley.edu/summer-2025/"
    url = "https://stat20.berkeley.edu/summer-2025/1-questions-and-data/01-understanding-the-world/notes.html" 
    content = "lecture content"
    documents = []

    # Iterate until there is no next page
    while True:
        response = requests.get(url)
        html_content = response.content
        soup = BeautifulSoup(html_content, "html.parser")
        main = soup.find("main", id = "quarto-document-content")
        header = soup.find("header", id ="quarto-header")

        if main:  
            content = clean_text(main.get_text())

        if header:
            title = header.find("h1", class_ = "quarto-secondary-nav-title no-breadcrumbs").get_text()

        if content:
            documents.append(
                Document(
                    page_content = content,
                    metadata = {
                        "url": url,
                        "title": title
                    }
                )
            )

        next_page = soup.find("div", class_ = "nav-page nav-page-next")
        if next_page:
            next_href = next_page.find("a", href = True)
            if not next_href:
                break
            url = rel_url + clean_link(next_page.find("a", href = True)["href"])
        else:
            break
        
    return documents