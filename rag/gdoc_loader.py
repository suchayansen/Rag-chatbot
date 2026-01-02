import requests
import re
from bs4 import BeautifulSoup

def load_google_doc(url: str) -> str:
    # Extract document ID
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Invalid Google Docs link format.")

    doc_id = match.group(1)

    # Google Docs HTML export URL
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=html"

    response = requests.get(export_url)

    if response.status_code == 403:
        raise PermissionError(
            "Document is private. Please set it to 'Anyone with the link can view'."
        )

    if response.status_code != 200:
        raise ConnectionError("Unable to fetch the document.")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Extract full document text
    text = "\n".join(
        el.get_text(strip=True)
        for el in soup.find_all(["h1", "h2", "h3", "p", "li"])
        if el.get_text(strip=True)
    )

    if not text or len(text.split()) < 30:
        raise ValueError("The document is empty or too short.")

    return text
