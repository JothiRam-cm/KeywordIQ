import requests
from bs4 import BeautifulSoup

from search import get_google_results

def scrape_page_details(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(res.text, "html.parser")

    # 🏷️ Title or H1
    title = soup.title.string.strip() if soup.title else None
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None

    # 📝 Paragraphs
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text_content = " ".join(paragraphs)

    # 🔑 Meta Keywords
    meta_tag = soup.find("meta", attrs={"name": "keywords" , 'name':'description'})
    meta_keywords = meta_tag["content"].strip() if meta_tag and "content" in meta_tag.attrs else None

    return {
        "url": url,
        "title": title,
        "keywords": meta_keywords,
        "content": text_content  # limit to 1000 chars for readability
    }


if __name__ == "__main__":
    # Example: feed your URLs here
    res = get_google_results("artificial intelligence", num_results=2)
    urls = [i['link'] for i in res]
    print("🔗 URLs to scrape:", urls)
    for url in urls:
        data = scrape_page_details(url)
        if data:
            print(f"\n=== {data['url']} ===")
            print(f"Title: {data['title']}")
            print(f"Keywords: {data['keywords']}")
            print(f"Content: {data['content'][:300]}...\n")