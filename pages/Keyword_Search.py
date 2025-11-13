import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime
import json
import re
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from search import get_google_results
from scraper import scrape_page_details
from nlp_kewords import process_text_chunk, extract_keywords_keybert

# --- Page Configuration ---
st.set_page_config(
    page_title="KeywordIQ - Smart Search & Extraction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .keyword-tag {
        display: inline-block;
        background: #e3f2fd;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def validate_query(query: str) -> tuple[bool, str]:
    """Validate search query"""
    if not query.strip():
        return False, "Query cannot be empty"
    if len(query.strip()) < 2:
        return False, "Query too short (minimum 2 characters)"
    if len(query.strip()) > 200:
        return False, "Query too long (maximum 200 characters)"
    return True, ""

def process_single_result(res: Dict, method: str, index: int) -> Optional[Dict]:
    """Process a single search result with error handling"""
    try:
        title = res.get("title", "Untitled")
        url = res.get("link", "")
        snippet = res.get("snippet", "")
        
        # Scrape page
        page = scrape_page_details(url)
        if not page:
            return {
                "index": index,
                "url": url,
                "title": title,
                "snippet": snippet,
                "status": "failed",
                "error": "Scraping failed"
            }
        
        # Extract keywords
        text = page.get("content", "")
        if method == "RAKE":
            keywords = process_text_chunk(text) if text else []
        else:
            keywords = extract_keywords_keybert(text, top_n=10) if text else []
        
        return {
            "index": index,
            "url": url,
            "title": title,
            "snippet": snippet,
            "page_title": page.get("title", "N/A"),
            "meta_keywords": page.get("keywords", "N/A"),
            "content": text,
            "keywords": keywords,
            "status": "success",
            "word_count": len(text.split()) if text else 0
        }
    except Exception as e:
        return {
            "index": index,
            "url": res.get("link", ""),
            "title": res.get("title", "Untitled"),
            "snippet": res.get("snippet", ""),
            "status": "error",
            "error": str(e)
        }

def export_to_json(data: List[Dict]) -> str:
    """Export data to JSON format"""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "results": data
    }
    return json.dumps(export_data, indent=2)

# --- Initialize Session State ---
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "keyword_data" not in st.session_state:
    st.session_state.keyword_data = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🔍 KeywordIQ — Smart Search & Extraction</h1>
    <p>Advanced web scraping with AI-powered keyword extraction and analytics</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Search settings
    query = st.text_input(
        "Search Keyword:",
        value=st.session_state.current_query,
        placeholder="Enter your search term..."
    )
    
    num_results = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        help="More results = longer processing time"
    )
    
    method = st.radio(
        "Extraction Method",
        ["RAKE", "KeyBERT"],
        help="RAKE: Fast, rule-based | KeyBERT: AI-powered, context-aware"
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        parallel_processing = st.checkbox("Parallel Processing", value=True, help="Process multiple pages simultaneously")
        max_workers = st.slider("Max Workers", 1, 5, 3) if parallel_processing else 1
        show_preview = st.checkbox("Show Site Preview", value=False, help="Embed website preview (slower)")
        content_preview_length = st.slider("Content Preview Length", 100, 500, 300)
    
    st.divider()
    run_search = st.button("🚀 Run Search", type="primary", use_container_width=True)
    
    # Search history
    if st.session_state.search_history:
        st.subheader("📜 Recent Searches")
        for i, hist in enumerate(st.session_state.search_history[-5:]):
            if st.button(f"🔍 {hist}", key=f"hist_{i}", use_container_width=True):
                st.session_state.current_query = hist
                st.rerun()

# --- Main Content ---
if run_search:
    # Validate query
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        st.stop()
    
    # Add to history
    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)
    
    # Fetch search results
    with st.spinner("🔎 Searching Google..."):
        results = get_google_results(query, num_results=num_results)
    
    if not results:
        st.error("❌ No search results found. Please check your API configuration or try a different query.")
        st.stop()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Results Found", len(results))
    with col2:
        st.metric("Extraction Method", method)
    with col3:
        st.metric("Processing Mode", "Parallel" if parallel_processing else "Sequential")
    with col4:
        st.metric("Status", "Processing...")
    
    st.divider()
    st.subheader(f"📊 Results for: **{query}**")
    
    # Process results
    all_data = []
    failed_count = 0
    start_time = datetime.now()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if parallel_processing:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_result, res, method, i): i 
                for i, res in enumerate(results)
            }
            
            for completed_count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                all_data.append(result)
                
                progress = completed_count / len(results)
                progress_bar.progress(progress)
                status_text.text(f"Processed {completed_count}/{len(results)} pages...")
    else:
        # Sequential processing
        for i, res in enumerate(results):
            result = process_single_result(res, method, i)
            all_data.append(result)
            
            progress = (i + 1) / len(results)
            progress_bar.progress(progress)
            status_text.text(f"Processing {i+1}/{len(results)}...")
    
    # Sort by index
    all_data.sort(key=lambda x: x["index"])
    
    # Calculate stats
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    success_count = sum(1 for d in all_data if d["status"] == "success")
    failed_count = len(all_data) - success_count
    
    status_text.empty()
    progress_bar.empty()
    
    # Update metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Successful", success_count)
    with col2:
        st.metric("❌ Failed", failed_count)
    with col3:
        st.metric("⏱️ Time", f"{processing_time:.2f}s")
    with col4:
        st.metric("📈 Success Rate", f"{(success_count/len(all_data)*100):.1f}%")
    
    st.divider()
    
    # Display results
    tabs = st.tabs(["📄 Detailed Results", "📊 Data Table", "📈 Analytics"])
    
    with tabs[0]:
        for data in all_data:
            with st.expander(f"{data['index']+1}. {data['title']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**🔗 URL:** [{data['url']}]({data['url']})")
                    st.caption(data['snippet'])
                    
                    if data['status'] == "success":
                        st.markdown(f"**📄 Page Title:** {data['page_title']}")
                        st.markdown(f"**🏷️ Meta Keywords:** {data['meta_keywords']}")
                        st.markdown(f"**📝 Word Count:** {data['word_count']:,}")
                        
                        # Content preview
                        st.markdown("**Content Preview:**")
                        preview = data['content'][:content_preview_length]
                        st.text_area("", preview, height=100, key=f"content_{data['index']}", disabled=True)
                        
                        # Keywords
                        st.markdown("**🧠 Extracted Keywords:**")
                        if data['keywords']:
                            keywords_html = "".join([
                                f'<span class="keyword-tag">{kw}</span>' 
                                for kw in data['keywords']
                            ])
                            st.markdown(keywords_html, unsafe_allow_html=True)
                        else:
                            st.info("No keywords extracted")
                    else:
                        st.error(f"⚠️ Error: {data.get('error', 'Unknown error')}")
                
                with col2:
                    st.markdown("**Status**")
                    if data['status'] == "success":
                        st.success("✅ Success")
                    else:
                        st.error("❌ Failed")
                
                # Site preview (optional)
                if show_preview and data['status'] == "success":
                    st.markdown("#### 🌐 Site Preview:")
                    try:
                        components.iframe(data['url'], height=400, scrolling=True)
                    except:
                        st.info(f"Preview unavailable. [Open externally]({data['url']})")
    
    with tabs[1]:
        # Create DataFrame
        df_data = []
        for d in all_data:
            if d['status'] == "success":
                df_data.append({
                    "URL": d['url'],
                    "Title": d['title'],
                    "Page Title": d['page_title'],
                    "Word Count": d['word_count'],
                    "Keywords": ", ".join(d['keywords']) if d['keywords'] else "None",
                    "Snippet": d['snippet']
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, height=400)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    f"keywordiq_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_data = export_to_json(all_data)
                st.download_button(
                    "⬇️ Download JSON",
                    json_data,
                    f"keywordiq_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                excel_buffer = pd.ExcelWriter(f"temp.xlsx", engine='xlsxwriter')
                df.to_excel(excel_buffer, index=False, sheet_name='Results')
                excel_buffer.close()
            
            # Save to session
            st.session_state.keyword_data = df
        else:
            st.warning("No successful results to display")
    
    with tabs[2]:
        if success_count > 0:
            # Analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Keyword Frequency")
                all_keywords = []
                for d in all_data:
                    if d['status'] == "success" and d['keywords']:
                        all_keywords.extend(d['keywords'])
                
                if all_keywords:
                    keyword_counts = pd.Series(all_keywords).value_counts().head(20)
                    st.bar_chart(keyword_counts)
                else:
                    st.info("No keywords to analyze")
            
            with col2:
                st.subheader("📈 Word Count Distribution")
                word_counts = [d['word_count'] for d in all_data if d['status'] == "success"]
                if word_counts:
                    wc_df = pd.DataFrame({"Word Count": word_counts})
                    st.line_chart(wc_df)
                else:
                    st.info("No data to visualize")
        else:
            st.warning("No successful results for analytics")

else:
    # Welcome screen
    st.info("👈 Configure your search in the sidebar and click **🚀 Run Search** to begin")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🔍 Smart Search
        - Google-powered results
        - Configurable result count
        - Search history tracking
        """)
    with col2:
        st.markdown("""
        ### 🧠 AI Extraction
        - RAKE algorithm
        - KeyBERT neural model
        - Automatic keyword ranking
        """)
    with col3:
        st.markdown("""
        ### 📊 Analytics
        - Data visualization
        - Export to CSV/JSON
        - Detailed metrics
        """)

# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# from search import get_google_results
# from scraper import scrape_page_details
# from nlp_kewords import process_text_chunk, extract_keywords_keybert

# st.set_page_config(page_title="Keyword Search", page_icon="🔍", layout="wide")

# st.title("🔍 KeywordIQ — Smart Search & Extraction")
# st.markdown("### Search topics, scrape pages, and extract key concepts automatically.")

# # --- Sidebar controls ---
# st.sidebar.header("⚙️ Configuration")
# query = st.sidebar.text_input("Enter search keyword:")
# num_results = st.sidebar.slider("Number of results to fetch", 1, 10, 5)
# method = st.sidebar.radio("Keyword Extraction Method", ["RAKE", "KeyBERT"])
# run_search = st.sidebar.button("Run Search 🔎")

# # --- Main Logic ---
# if run_search and query.strip():
#     with st.spinner("Fetching Google search results..."):
#         results = get_google_results(query, num_results=num_results)

#     if not results:
#         st.error("❌ No search results found. Check your API key or query.")
#     else:
#         st.success(f"✅ Found {len(results)} results for '{query}'")
#         st.subheader(f"Search Results for: **{query}**")

#         all_data = []
#         progress = st.progress(0)

#         for i, res in enumerate(results):
#             title = res.get("title", "Untitled")
#             url = res.get("link", "")
#             snippet = res.get("snippet", "")

#             with st.expander(f"{i+1}. {title}"):
#                 st.write(f"🔗 [Visit Website]({url})")
#                 st.caption(snippet)

#                 # --- Scrape the page ---
#                 page = scrape_page_details(url)
#                 if not page:
#                     st.warning("Failed to scrape content.")
#                     continue

#                 st.write(f"**Scraped Title:** {page['title'] or 'N/A'}")
#                 st.write(f"**Meta Keywords:** {page['keywords'] or 'N/A'}")
#                 st.write(f"**Content Preview:** {page['content'][:300]}...")

#                 # --- Keyword Extraction ---
#                 text = page["content"]
#                 if method == "RAKE":
#                     keywords = process_text_chunk(text)
#                 else:
#                     keywords = extract_keywords_keybert(text, top_n=10)

#                 st.markdown("**🧠 Extracted Keywords:**")
#                 st.write(", ".join(keywords))

#                 # --- Embedded Site Preview ---
#                 st.markdown("#### 🌐 Site Preview:")
#                 try:
#                     components.iframe(url, height=400, scrolling=True)
#                 except:
#                     st.info(f"[Open site externally]({url})")

#                 # --- Collect data ---
#                 all_data.append({
#                     "URL": url,
#                     "Title": title,
#                     "Snippet": snippet,
#                     "Keywords": ", ".join(keywords)
#                 })

#             progress.progress((i + 1) / len(results))

#         # --- Display and Save ---
#         if all_data:
#             df = pd.DataFrame(all_data)
#             st.subheader("📊 Extracted Data Summary")
#             st.dataframe(df)

#             csv = df.to_csv(index=False).encode("utf-8")
#             st.download_button(
#                 "⬇️ Download Extracted Data (CSV)",
#                 csv,
#                 "keywordiq_results.csv",
#                 "text/csv",
#                 key="download-csv",
#             )

#             # Save to session for analytics page
#             st.session_state["keyword_data"] = df

# else:
#     st.info("💡 Enter a keyword in the sidebar and click **Run Search 🔎** to begin.")
