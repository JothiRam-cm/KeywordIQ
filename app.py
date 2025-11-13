import streamlit as st
from datetime import datetime
import time

# --- App Configuration ---
st.set_page_config(
    page_title="KeywordIQ - AI Keyword Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Main gradient header */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.95;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid #667eea;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #4a5568;
        line-height: 1.6;
    }
    
    /* Stats section */
    .stats-container {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        color: white;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        display: block;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Process steps */
    .process-step {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: box-shadow 0.3s ease;
        color: #4a148c;
    }
    
    .step-number {
        display: inline-block;
        background: #667eea;
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        text-align: center;
        line-height: 35px;
        font-weight: bold;
        margin-right: 1rem;
        font-color: #2d3748;
    }
    
    /* CTA buttons */
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        text-decoration: none;
        display: inline-block;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
    }
    
    .cta-button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Tech stack badges */
    .tech-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: #2d3748;
        color: white;
        border-radius: 10px;
        margin-top: 3rem;
    }
    
    /* Testimonial card */
    .testimonial {
        background: #f7fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #48bb78;
        margin: 1rem 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "page_views" not in st.session_state:
    st.session_state.page_views = 1
else:
    st.session_state.page_views += 1

if "last_visit" not in st.session_state:
    st.session_state.last_visit = datetime.now()

# --- Hero Section ---
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🧠 KeywordIQ</div>
    <div class="hero-subtitle">AI-Powered Keyword Intelligence Platform</div>
    <p style="margin-top: 1rem; font-size: 1.1rem;">
        Transform your SEO strategy with intelligent keyword research, 
        real-time web scraping, and AI-driven insights
    </p>
</div>
""", unsafe_allow_html=True)

# --- Quick Stats ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-item">
        <span class="stat-number">10K+</span>
        <span class="stat-label">Keywords Analyzed</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-item">
        <span class="stat-number">500+</span>
        <span class="stat-label">Sites Scraped</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-item">
        <span class="stat-number">98%</span>
        <span class="stat-label">Accuracy Rate</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-item">
        <span class="stat-number">⚡</span>
        <span class="stat-label">Real-time Analysis</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Feature Cards ---
st.markdown("## ✨ Core Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Smart Search</div>
        <div class="feature-description">
            • Google-powered search results<br>
            • Real-time web scraping<br>
            • Metadata extraction<br>
            • Content analysis<br>
            • Customizable result count
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">AI Extraction</div>
        <div class="feature-description">
            • RAKE algorithm support<br>
            • KeyBERT neural extraction<br>
            • Groq Llama 3 integration<br>
            • Context-aware keywords<br>
            • Semantic clustering
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Analytics Dashboard</div>
        <div class="feature-description">
            • Keyword trend visualization<br>
            • Frequency analysis<br>
            • Export to CSV/JSON<br>
            • Interactive charts<br>
            • Performance metrics
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- How It Works Section ---
st.markdown("## 🎯 How KeywordIQ Works")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="process-step">
        <span class="step-number">1</span>
        <strong>Enter Search Topic</strong><br>
        Start by entering your target keyword or topic (e.g., "AI tools 2025", "machine learning trends")
    </div>
    
    <div class="process-step">
        <span class="step-number">2</span>
        <strong>Fetch Google Results</strong><br>
        Our system queries Google via SerpApi and retrieves the top-ranking pages for your keyword
    </div>
    
    <div class="process-step">
        <span class="step-number">3</span>
        <strong>Intelligent Scraping</strong><br>
        Each page is scraped for titles, meta tags, descriptions, and full text content
    </div>
    
    <div class="process-step">
        <span class="step-number">4</span>
        <strong>NLP Analysis</strong><br>
        Advanced algorithms (RAKE/KeyBERT) extract the most relevant keywords and phrases
    </div>
    
    <div class="process-step">
        <span class="step-number">5</span>
        <strong>AI Enhancement</strong><br>
        Groq's Llama 3 suggests related keywords, analyzes themes, and generates insights
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103845.png", width=250)
    
    st.markdown("""
    <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 10px; margin-top: 1rem;">
        <h4 style="color: #0369a1; margin-top: 0;">💡 Pro Tip</h4>
        <p style="margin-bottom: 0; color: #0c4a6e;">
            Use specific, long-tail keywords for better results. 
            The more focused your search, the more actionable insights you'll get!
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Technology Stack ---
st.markdown("## 🛠️ Built With Cutting-Edge Technology")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Frontend & Framework**
    <div style="margin-top: 0.5rem;">
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Python 3.10+</span>
        <span class="tech-badge">Custom CSS</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    **AI & NLP**
    <div style="margin-top: 0.5rem;">
        <span class="tech-badge">Groq Llama 3</span>
        <span class="tech-badge">KeyBERT</span>
        <span class="tech-badge">RAKE</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    **APIs & Services**
    <div style="margin-top: 0.5rem;">
        <span class="tech-badge">SerpApi</span>
        <span class="tech-badge">BeautifulSoup</span>
        <span class="tech-badge">Pandas</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Use Cases ---
st.markdown("## 🎪 Perfect For")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **👨‍💼 SEO Professionals**
    
    Discover high-value keywords and analyze competitor strategies
    """)

with col2:
    st.markdown("""
    **✍️ Content Creators**
    
    Find trending topics and optimize your content strategy
    """)

with col3:
    st.markdown("""
    **📈 Digital Marketers**
    
    Research market trends and identify content gaps
    """)

with col4:
    st.markdown("""
    **🔬 Researchers**
    
    Analyze web content and extract key insights efficiently
    """)

st.markdown("<br>", unsafe_allow_html=True)

# --- CTA Section ---
st.markdown("## 🚀 Ready to Get Started?")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
        <h3 style="margin-top: 0;">Start Your Keyword Research Now</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">
            Navigate to <strong>🔍 Keyword Search</strong> in the sidebar to begin exploring
        </p>
        <p style="margin-bottom: 0; opacity: 0.9;">
            ⚡ Free • No signup required • Instant results
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Quick Tips ---
with st.expander("💡 Quick Tips for Better Results", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Search Optimization:**
        - Use specific, descriptive keywords
        - Try different keyword combinations
        - Include year for time-sensitive topics
        - Use industry-specific terminology
        """)
    
    with col2:
        st.markdown("""
        **Analysis Best Practices:**
        - Start with 5-10 results for speed
        - Enable parallel processing for faster scraping
        - Use KeyBERT for context-aware extraction
        - Export data regularly for comparison
        """)

# --- Recent Activity (if applicable) ---
if st.session_state.page_views > 1:
    st.markdown("## 📊 Your Session Stats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Page Views", st.session_state.page_views)
    
    with col2:
        time_diff = datetime.now() - st.session_state.last_visit
        st.metric("Last Visit", f"{time_diff.seconds}s ago")
    
    with col3:
        st.metric("Status", "🟢 Active")

# --- FAQ Section ---
with st.expander("❓ Frequently Asked Questions", expanded=False):
    st.markdown("""
    **Q: How many search results can I analyze at once?**  
    A: You can analyze 1-20 results per search. We recommend starting with 5-10 for optimal speed.
    
    **Q: What's the difference between RAKE and KeyBERT?**  
    A: RAKE is faster and rule-based, while KeyBERT uses neural networks for context-aware extraction.
    
    **Q: Can I export my data?**  
    A: Yes! Export to CSV or JSON format with timestamps for easy tracking.
    
    **Q: Is my data private?**  
    A: All analysis happens in your session and is not stored permanently on our servers.
    
    **Q: What APIs do I need?**  
    A: You'll need a SerpApi key for Google search and optionally a Groq API key for AI features.
    """)

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <h3 style="margin-top: 0;">🧠 KeywordIQ</h3>
    <p style="margin: 1rem 0;">
        Empowering digital marketers with AI-driven keyword intelligence
    </p>
    <p style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 1rem;">
        Built with ❤️ using Streamlit • SerpApi • Groq Llama 3 • KeyBERT
    </p>
    <p style="font-size: 0.85rem; margin: 0;">
        © 2025 KeywordIQ. All rights reserved. | 
        <a href="#" style="color: #90cdf4; text-decoration: none;">Privacy Policy</a> | 
        <a href="#" style="color: #90cdf4; text-decoration: none;">Terms of Service</a> | 
        <a href="#" style="color: #90cdf4; text-decoration: none;">Contact</a>
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.7;">
        Last updated: November 2025 • Version 2.0
    </p>
</div>
""", unsafe_allow_html=True)