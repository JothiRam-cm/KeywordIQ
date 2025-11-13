import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from generator import summarize_keyword_themes
import warnings
from PIL import Image
import numpy as np
from itertools import combinations

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Suppress large image warnings ---
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Keyword Analytics", page_icon="📊", layout="wide")
st.title("📊 KeywordIQ — Enhanced Keyword Analytics")
st.caption("Analyze, visualize, and understand keyword patterns intelligently.")

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
st.sidebar.header("📂 Load Keyword Data")
uploaded_file = st.sidebar.file_uploader("Upload keywordiq_results.csv", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo data")

# SAMPLE DATA WHEN DEMO MODE IS ON
sample_df = pd.DataFrame({
    "Title": [
        "AI Tools for Business", "Machine Learning Basics", "Neural Networks Deep Dive",
        "AI Automation Guide", "Deep Learning Tutorial", "ML Algorithms Overview",
        "Productivity with AI", "Neural Network Architecture", "Supervised Learning Methods",
        "AI Tools Comparison"
    ],
    "Keywords": [
        "ai tools, automation, productivity, machine learning, business intelligence",
        "machine learning, supervised learning, algorithms, regression, classification",
        "neural networks, deep learning, artificial intelligence, backpropagation",
        "ai automation, workflow optimization, productivity tools, machine learning",
        "deep learning, neural networks, computer vision, natural language processing",
        "machine learning algorithms, decision trees, random forest, svm",
        "productivity, automation tools, ai assistants, workflow management",
        "neural networks, architecture design, layers, activation functions",
        "supervised learning, training data, model evaluation, accuracy",
        "ai tools, software comparison, machine learning platforms, cloud ai"
    ]
})

# LOAD FINAL DATAFRAME
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_demo:
    df = sample_df
else:
    st.info("📥 Upload your `keywordiq_results.csv` or enable demo mode to view analytics.")
    st.stop()

st.success("✅ Keyword data loaded successfully!")

# ---------------------------------------------------------------
# PREPROCESS KEYWORDS
# ---------------------------------------------------------------
all_keywords = []
keyword_per_title = {}

for idx, row in df.iterrows():
    raw_kw = str(row.get("Keywords", ""))
    if raw_kw.strip():
        keywords = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]
    else:
        keywords = []
    all_keywords.extend(keywords)
    keyword_per_title[row["Title"]] = keywords

# FREQUENCY STATS
freq = Counter(all_keywords)
unique_keywords = list(freq.keys())
total_count = len(all_keywords)
unique_count = len(unique_keywords)
diversity_index = round(unique_count / total_count, 2) if total_count else 0
focus_score = round(sum([v for _, v in freq.most_common(10)]) / total_count, 2) if total_count else 0
avg_length = round(sum(len(k.split()) for k in unique_keywords) / unique_count, 2) if unique_count else 0

# PRECOMPUTE co-occurrence matrix
cooccurrence = Counter()
for kws in keyword_per_title.values():
    for pair in combinations(sorted(set(kws)), 2):
        cooccurrence[pair] += 1

single_use = [(kw, count) for kw, count in freq.items() if count == 1]

# ---------------------------------------------------------------
# SIDEBAR FILTERS (FIXED SLIDER BUG)
# ---------------------------------------------------------------
st.sidebar.header("🔍 Filters & Settings")

max_freq_val = max(freq.values()) if freq else 1

if max_freq_val == 1:
    min_freq = 1
    st.sidebar.info("⚠️ All keywords appear only once — filtering disabled.")
else:
    min_freq = st.sidebar.slider(
        "Minimum keyword frequency",
        min_value=1,
        max_value=max_freq_val,
        value=1
    )

top_n = st.sidebar.slider("Top N keywords to display", 5, 50, 20)

# ---------------------------------------------------------------
# METRIC DASHBOARD
# ---------------------------------------------------------------
st.subheader("🧾 Keyword Summary Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Keywords", total_count)
col2.metric("Unique Keywords", unique_count)
col3.metric("Diversity Index", diversity_index)
col4.metric("Focus Score", focus_score)
col5.metric("Avg Keyword Length", f"{avg_length} words")

if focus_score > 0.6:
    st.info("💡 Highly focused — your keywords center around a narrow topic.")
elif focus_score > 0.4:
    st.info("💡 Moderately focused — a balanced distribution across subtopics.")
else:
    st.info("💡 Broad — keywords are spread across multiple themes.")

# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🔗 Relationships", "📈 Trends", "🎯 Keyword Gap", "💡 Insights", "🧠 Semantic Clustering", "🔍 Keyword Expansion Engine"
])

# ================================
# TAB 1: OVERVIEW
# ================================
with tab1:
    st.subheader("🎨 Keyword Visual Analytics")

    col1, col2 = st.columns(2)

    # WORD CLOUD
    with col2:
        st.markdown("#### ☁️ Word Cloud")
        text_input = " ".join(all_keywords)[:40000]
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            stopwords=STOPWORDS,
            max_words=200,
            colormap="viridis"
        ).generate(text_input)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # BAR CHART
    with col1:
        st.markdown(f"#### 📈 Top {top_n} Keywords")
        top_kw = freq.most_common(top_n)
        if top_kw:
            labels, values = zip(*top_kw)
            colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.barh(labels, values, color=colors)
            ax2.invert_yaxis()
            ax2.set_xlabel("Frequency")
            st.pyplot(fig2)
        else:
            st.warning("No keywords available for plotting.")

    # TREEMAP
    st.markdown("#### 🧩 Keyword Share Treemap")
    treemap_df = pd.DataFrame(top_kw, columns=["Keyword", "Frequency"])
    fig3 = px.treemap(
        treemap_df, path=["Keyword"], values="Frequency",
        color="Frequency", color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig3, width="stretch")

# ================================
# TAB 2: RELATIONSHIPS
# ================================
with tab2:
    st.subheader("🔗 Co-occurrence & Keyword Relationships")

    if cooccurrence:
        top_pairs = cooccurrence.most_common(15)
        st.markdown("#### 📋 Top Co-occurring Pairs")
        for pair, count in top_pairs[:10]:
            st.markdown(f"• **{pair[0]}** ↔ **{pair[1]}** ({count}x)")
    else:
        st.info("Not enough data for co-occurrence patterns.")

# ================================
# TAB 3: TRENDS & COMPLEXITY
# ================================
with tab3:
    st.subheader("📈 Keyword Trends & Complexity")

    keyword_lengths = [len(k.split()) for k in unique_keywords]
    char_lengths = [len(k) for k in unique_keywords]

    col1, col2 = st.columns(2)

    with col1:
        length_df = pd.DataFrame(keyword_lengths, columns=["Words"])
        fig7 = px.histogram(length_df, x="Words", nbins=10,
                            title="Keyword Length Distribution")
        st.plotly_chart(fig7, width="stretch")

    with col2:
        fig8 = px.box(x=char_lengths, title="Keyword Character Length Boxplot")
        st.plotly_chart(fig8, width="stretch")

# ================================
# TAB 4: KEYWORD GAP ANALYSIS
# ================================
with tab4:
    st.subheader("🎯 Keyword Gap Analysis")

    mid_freq = [(kw, c) for kw, c in freq.items() if 2 <= c <= 4]

    st.markdown("#### 🌱 Emerging Keywords (2–4 uses)")
    if mid_freq:
        df_mid = pd.DataFrame(mid_freq, columns=["Keyword", "Count"])
        st.dataframe(df_mid.head(15))
    else:
        st.info("No emerging keywords found.")

# ================================
# TAB 5: AI INSIGHTS
# ================================
with tab5:
    st.subheader("💡 AI Insights & Recommendations")

    freq_list = list(freq.values())
    st.markdown("#### 📈 Statistical Summary")
    col1, col2, col3 = st.columns(3)
    col1.write(f"Mean: {np.mean(freq_list):.2f}")
    col2.write(f"Median: {np.median(freq_list):.2f}")
    col3.write(f"Std Dev: {np.std(freq_list):.2f}")

    st.markdown("---")
    st.markdown("#### 🧠 AI Keyword Theme Summary")

    if st.button("Generate AI Theme Analysis"):
        with st.spinner("Groq Llama analyzing themes..."):
            try:
                ai_summary = summarize_keyword_themes(unique_keywords[:100])
                st.success("Analysis complete!")
                st.markdown(ai_summary)
            except Exception as e:
                st.error(f"Groq Error: {e}")
# ---------------------------------------------------------------
# TAB 6: SEMANTIC CLUSTERING (PHASE 2)
# ---------------------------------------------------------------

with tab6:
    st.subheader("🧠 AI Semantic Keyword Clustering")
    st.caption("Keywords are grouped using embeddings + KMeans, then labeled using Groq Llama.")

    if st.button("Generate Semantic Clusters"):
        with st.spinner("Generating embeddings and clustering keywords..."):
            try:
                # Load embedding model
                model = SentenceTransformer('all-MiniLM-L6-v2')

                # Compute embeddings
                embeddings = model.encode(unique_keywords)

                # Determine optimal number of clusters
                k = max(2, int(len(unique_keywords) ** 0.5 / 1.5))

                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(embeddings)

                # Dimensionality reduction for visualization
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(embeddings)

                cluster_df = pd.DataFrame({
                    "Keyword": unique_keywords,
                    "Cluster": labels,
                    "X": reduced[:, 0],
                    "Y": reduced[:, 1]
                })

                st.success(f"Generated {k} meaningful clusters!")

                # Plot clusters
                fig = px.scatter(
                    cluster_df,
                    x="X", y="Y",
                    color="Cluster",
                    hover_data=["Keyword"],
                    title="Semantic Keyword Clusters (PCA Projection)",
                    color_continuous_scale="Plasma",
                    width=900, height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                # AI-generated cluster themes
                st.markdown("### 🧩 Cluster Theme Labels (Groq Llama)")

                theme_results = {}
                for c in sorted(cluster_df["Cluster"].unique()):
                    subset = cluster_df[cluster_df["Cluster"] == c]["Keyword"].tolist()
                    prompt = f"""
Analyze the following keywords and provide:

1. A short 3–5 word cluster theme title
2. One-sentence description of what the cluster represents

Keywords:
{subset}
"""

                    theme_results[c] = summarize_keyword_themes(subset)

                for cluster_id, theme_text in theme_results.items():
                    st.markdown(f"#### Cluster {cluster_id}")
                    st.markdown(theme_text)
                    st.markdown("---")

                # Export option
                csv_data = cluster_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Cluster Results",
                    csv_data,
                    "semantic_clusters.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")
                st.info("Make sure you installed: sentence-transformers, sklearn")

# ---------------------------------------------------------------
# TAB 7: KEYWORD EXPANSION ENGINE (PHASE 3)
# ---------------------------------------------------------------
with tab7:
    st.subheader("🔍 AI-Powered Keyword Expansion Engine")
    st.caption("Generate new keyword ideas, compare with your dataset, and identify opportunities.")

    # -------------------------------
    # INPUTS
    # -------------------------------
    main_topic = st.text_input(
        "Enter your main topic:",
        placeholder="e.g., AI automation tools for business"
    )

    num_keywords = st.slider(
        "Number of keywords to generate:",
        10, 100, 30
    )

    depth = st.selectbox(
        "Expansion depth:",
        ["Broad Concepts", "Specific / Niche", "Mixed"]
    )

    generate_btn = st.button("Generate Expanded Keywords")

    if generate_btn:
        if not main_topic.strip():
            st.error("Please enter a topic.")
            st.stop()

        with st.spinner("Generating keyword expansion with Groq Llama..."):
            try:
                # AI Generator
                prompt = f"""
You are an SEO keyword strategist.

Generate {num_keywords} keyword ideas related to:

"{main_topic}"

Depth style: {depth}

Rules:
- Output ONLY a list of keywords
- No numbering, no paragraphs
- One keyword per line
- Avoid duplicates
"""

                generated_text = summarize_keyword_themes([prompt])

                # Clean into list
                new_keywords = [
                    k.strip().lower()
                    for k in generated_text.split("\n")
                    if k.strip()
                ]

                new_keywords = list(set(new_keywords))

                st.success("✅ Keyword expansion generated!")

                # -------------------------------
                # Comparative Analysis
                # -------------------------------

                st.markdown("### 🆚 Comparison with Your Current Keywords")

                existing_keywords = set(unique_keywords)
                generated_set = set(new_keywords)

                missing_keywords = list(generated_set - existing_keywords)
                overlapping_keywords = list(generated_set & existing_keywords)

                col1, col2, col3 = st.columns(3)
                col1.metric("Generated", len(new_keywords))
                col2.metric("New Opportunities", len(missing_keywords))
                col3.metric("Already Covered", len(overlapping_keywords))

                # Show new opportunity keywords
                st.markdown("#### 🎯 New Opportunity Keywords")
                if missing_keywords:
                    for kw in sorted(missing_keywords):
                        st.markdown(f"• {kw}")
                else:
                    st.info("All generated keywords are already covered in your dataset.")

                # Show overlapping keywords
                st.markdown("#### 🔄 Keywords Already in Your Dataset")
                if overlapping_keywords:
                    for kw in sorted(overlapping_keywords):
                        st.markdown(f"• {kw}")
                else:
                    st.info("None of the generated keywords overlap.")

                # -------------------------------
                # Visual Analysis
                # -------------------------------
                st.markdown("### 📊 Frequency Comparison Chart")

                comp_df = pd.DataFrame({
                    "Keyword": list(generated_set),
                    "Type": ["existing" if kw in existing_keywords else "new" for kw in generated_set]
                })

                fig = px.bar(
                    comp_df,
                    x="Keyword",
                    color="Type",
                    title="Generated Keywords — New vs Existing",
                    color_discrete_map={"new": "blue", "existing": "gray"}
                )

                st.plotly_chart(fig, width="stretch")

                # -------------------------------
                # Export Options
                # -------------------------------
                csv_out = pd.DataFrame({
                    "Keyword": list(generated_set),
                    "Type": ["existing" if kw in existing_keywords else "new" for kw in generated_set]
                }).to_csv(index=False)

                st.download_button(
                    "📥 Download Expanded Keyword List",
                    csv_out,
                    "expanded_keywords.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Groq API error: {str(e)}")



# ---------------------------------------------------------------
# EXPORT OPTIONS
# ---------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("📥 Export Data")

export_df = pd.DataFrame(freq.most_common(), columns=["Keyword", "Frequency"])
export_df["Percentage"] = (export_df["Frequency"] / total_count * 100).round(2)

st.sidebar.download_button(
    "Download Keyword Report (CSV)",
    data=export_df.to_csv(index=False),
    file_name="keyword_report.csv",
    mime="text/csv"
)
