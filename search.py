from serpapi import GoogleSearch

# 🔑 Replace with your actual API key
api_key = "4a917b1bc35f18cb4322a45a3bac68a6d8428816232b58bb0f0b6b1f9c911905"

def get_google_results(query, num_results=5, location="India"):
    params = {
        "engine": "google",
        "q": query,
        "location": location,
        "num": num_results,
        "api_key": api_key,
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" not in results:
        print("No organic results found.")
        return []

    data = []
    for r in results["organic_results"]:
        item = {
            "position": r.get("position"),
            "title": r.get("title"),
            "link": r.get("link"),
            "snippet": r.get("snippet"),
        }
        data.append(item)
    
    return data


if __name__ == "__main__":
    keyword = input("🔍 Enter your keyword: ")
    search_results = get_google_results(keyword)

    print("\n=== 🔎 Top Google Results ===\n")
    for res in search_results:
        print(f"{res['position']}. {res['title']}")
        print(f"   URL: {res['link']}")
        print(f"   Snippet: {res['snippet']}\n")