import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

from rake_nltk import Rake
from scraper import scrape_page_details
from search import get_google_results

from keybert import KeyBERT
kw_model = KeyBERT()

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Preprocess text: tokenize, remove stopwords/punctuation, and lemmatize."""
    text = text.lower()
    tokens = word_tokenize(text)

    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words and len(token) > 2
    ]

    return " ".join(cleaned_tokens)


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

    sample_text = {
        i: f"{p.get('title','')}\n{p.get('keywords','')}\n{p.get('content','')}\n"
        for i, p in enumerate(pages)
    }

    print("\n🧩 Extracted Keywords:\n")
    for i, text in sample_text.items():
        print(f"\n--- From Document {i} ---")
        keywords = process_text_chunk(text)
        filtered_keywords = extract_keywords_keybert(keywords)
        print("KeyBERT Filtered Keywords:")
        for kw in filtered_keywords:
            print("-", kw)
