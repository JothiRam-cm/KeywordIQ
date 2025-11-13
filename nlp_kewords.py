import spacy
from rake_nltk import Rake

from scraper import scrape_page_details
from search import get_google_results

from keybert import KeyBERT
kw_model = KeyBERT()


# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Preprocess text: remove stopwords, punctuation, and lemmatize."""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha and len(token) > 2
    ]
    return " ".join(tokens)

def extract_keywords(text, top_n=15):
    """Extract top keywords and phrases using RAKE."""
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()[:top_n]
    return ranked_phrases

def process_text_chunk(text):
    """Clean text, process NLP, extract keywords."""
    cleaned = clean_text(text)
    keywords = extract_keywords(cleaned)
    return keywords


def extract_keywords_keybert(text, top_n=10):
    return [kw for kw, _ in kw_model.extract_keywords(text, top_n=top_n)]



if __name__ == "__main__":
    # Example usage — feed text from your scraper
    results = get_google_results("artificial intelligence", num_results=2)
    pages = []
    for r in results:
        details = scrape_page_details(r['link'])
        if details:
            pages.append(details)

    sample_text = {i: f"{p.get('title','')}\n{p.get('keywords','')}\n{p.get('content','')}\n" for i, p in enumerate(pages)}

    print("\n🧩 Extracted Keywords:\n")
    for i, text in sample_text.items():
        print(f"\n--- From Document {i} ---")
        keywords = process_text_chunk(text)
        filtered_keywords = extract_keywords_keybert(keywords)
        print("KeyBERT Filtered Keywords:")
        for kw in filtered_keywords:
            print("-", kw)
