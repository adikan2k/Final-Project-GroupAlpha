import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Scholarly Topic Navigator", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%); }
    h1, h2, h3 { color: #e94560 !important; }
    .stButton > button { background: linear-gradient(90deg, #e94560 0%, #0f3460 100%); color: white; border-radius: 25px; }
    .score-badge { 
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%); 
        padding: 8px 16px; 
        border-radius: 20px; 
        font-size: 1.2em; 
        font-weight: bold; 
        color: white; 
        display: inline-block;
        margin: 5px 0;
    }
    .paper-title { font-size: 1.1em; font-weight: 600; }
    .year-badge {
        background: #0f3460;
        padding: 4px 12px;
        border-radius: 15px;
        color: white;
        font-size: 0.9em;
        margin-right: 8px;
    }
    .cat-badge {
        background: #e94560;
        padding: 4px 12px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

def detect_environment():
    if Path("/kaggle").exists():
        env = "kaggle"
        working = Path("/kaggle/working")
    elif Path("/content").exists():
        env = "colab"
        working = Path("/content")
    else:
        env = "local"
        working = Path(".").resolve()
    
    search_paths = []
    if env == "kaggle":
        search_paths = ["/kaggle/working", "/kaggle/input"]
    elif env == "colab":
        search_paths = ["/content", "/content/drive/MyDrive"]
    else:
        search_paths = [str(working), str(working.parent)]
    
    parquet_files = []
    npy_files = []
    for search_path in search_paths:
        if Path(search_path).exists():
            parquet_files.extend(glob.glob(f"{search_path}/**/*.parquet", recursive=True))
            npy_files.extend(glob.glob(f"{search_path}/**/*.npy", recursive=True))
    
    return {"env": env, "working": working, "parquet_files": parquet_files, "npy_files": npy_files}

ENV_INFO = detect_environment()

class BM25Retriever:
    def __init__(self, corpus):
        self.bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
    def search(self, query, top_k=10):
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

class SemanticRetriever:
    def __init__(self, embeddings):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-9)
        if FAISS_AVAILABLE:
            import faiss
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            self.use_faiss = True
        else:
            self.use_faiss = False
    
    def search(self, query, top_k=10):
        qv = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        if self.use_faiss:
            scores, idx = self.index.search(qv, top_k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]
        else:
            scores = np.dot(self.embeddings, qv.T).flatten()
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [(int(i), float(scores[i])) for i in top_idx]

class HybridRetriever:
    def __init__(self, bm25, semantic):
        self.bm25 = bm25
        self.semantic = semantic
    def search(self, query, top_k=10):
        b_res = dict(self.bm25.search(query, 50))
        s_res = dict(self.semantic.search(query, 50))
        all_idx = set(b_res) | set(s_res)
        b_max = max(b_res.values()) if b_res else 1
        s_max = max(s_res.values()) if s_res else 1
        combined = [(i, 0.3*b_res.get(i,0)/(b_max+1e-9) + 0.7*s_res.get(i,0)/(s_max+1e-9)) for i in all_idx]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

def get_summary(text, n=3):
    if not text or len(text) < 50:
        return text
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) <= n:
        return text
    return ". ".join(sentences[:n]) + "."

@st.cache_resource
def load_data():
    pq_files = ENV_INFO["parquet_files"]
    if not pq_files:
        raise FileNotFoundError(f"No parquet files found. Searched: {ENV_INFO}")
    
    priority_order = ["papers_with_categories", "papers_fully_labeled", "papers_with_topics", "cleaned_papers"]
    selected_file = pq_files[0]
    for priority in priority_order:
        for pf in pq_files:
            if priority in pf:
                selected_file = pf
                break
    
    df = pd.read_parquet(selected_file)
    st.info(f"Loaded: {Path(selected_file).name} ({len(df):,} papers)")
    
    npy_files = ENV_INFO["npy_files"]
    emb = None
    for nf in npy_files:
        if "sbert" in nf.lower() or "embed" in nf.lower():
            emb = np.load(nf)
            break
    if emb is None and npy_files:
        emb = np.load(npy_files[0])
    if emb is None:
        st.warning("No embeddings found - using random vectors")
        emb = np.random.randn(len(df), 384).astype("float32")
    
    abs_col = next((c for c in ["original_abstract", "abstract", "text"] if c in df.columns), df.columns[1])
    corpus = (df["title"].fillna("") + " " + df[abs_col].fillna("")).tolist()
    bm25 = BM25Retriever(corpus)
    semantic = SemanticRetriever(emb)
    hybrid = HybridRetriever(bm25, semantic)
    return df, hybrid, abs_col

if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

def main():
    st.markdown("<h1 style='text-align:center; color:#e94560;'>Scholarly Topic Navigator</h1>", unsafe_allow_html=True)
    st.caption(f"Environment: {ENV_INFO['env']}")
    
    try:
        df, retriever, abs_col = load_data()
        st.success(f"Ready! {len(df):,} papers indexed")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.json(ENV_INFO)
        return
    
    # Calculate year range
    year_min, year_max = None, None
    if "year" in df.columns:
        valid_years = df["year"].dropna()
        if len(valid_years) > 0:
            year_min = int(valid_years.min())
            year_max = int(valid_years.max())
    
    with st.sidebar:
        st.markdown("## Dashboard")
        st.metric("Papers", f"{len(df):,}")
        if "category" in df.columns:
            st.metric("Categories", df["category"].nunique())
        if year_min and year_max:
            st.metric("Years", f"{year_min} - {year_max}")
        st.markdown("---")
        num_results = st.slider("Results", 3, 20, 5)
        page = st.radio("Navigate:", ["Search", "Analytics", "Visualizations"])
    
    if page == "Search":
        search_page(df, retriever, abs_col, num_results)
    elif page == "Analytics":
        analytics_page(df)
    else:
        viz_page()

def search_page(df, retriever, abs_col, num_results):
    st.markdown("## Search Papers")
    with st.form("search_form"):
        query = st.text_input("Enter query:", value=st.session_state.search_query, placeholder="e.g., transformer attention")
        col1, col2 = st.columns(2)
        with col1:
            search_btn = st.form_submit_button("Search", type="primary")
        with col2:
            random_btn = st.form_submit_button("Random")
    
    if random_btn:
        query = np.random.choice(["deep learning", "NLP", "attention mechanism", "neural network"])
        st.session_state.search_query = query
        st.rerun()
    if search_btn and query:
        st.session_state.search_query = query
        st.session_state.summaries = {}
    
    if st.session_state.search_query:
        query = st.session_state.search_query
        st.markdown(f"### Results for: *{query}*")
        results = retriever.search(query, num_results)
        
        for rank, (idx, score) in enumerate(results, 1):
            row = df.iloc[idx]
            abstract = str(row.get(abs_col, ""))
            title = str(row.get("title", "Untitled"))[:80]
            
            # Get year and category
            year_str = ""
            cat_str = ""
            if "year" in df.columns and pd.notna(row.get("year")):
                year_str = f"<span class='year-badge'>{int(row['year'])}</span>"
            if "category" in df.columns and pd.notna(row.get("category")):
                cat_str = f"<span class='cat-badge'>{row['category']}</span>"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #e94560;'>
                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                    <span class='paper-title' style='color: white; flex: 1;'>{rank}. {title}...</span>
                    <span class='score-badge'>Score: {score:.3f}</span>
                </div>
                <div style='margin-top: 8px;'>{year_str} {cat_str}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View Details", expanded=(rank==1)):
                st.markdown("#### Abstract")
                st.write(abstract[:500] + "..." if len(abstract) > 500 else abstract)
                
                key = f"sum_{idx}"
                if st.button("Generate Summary", key=f"btn_{idx}"):
                    st.session_state.summaries[key] = get_summary(abstract)
                if key in st.session_state.summaries:
                    st.success(f"**Summary:** {st.session_state.summaries[key]}")

def analytics_page(df):
    st.markdown("## Analytics Dashboard")
    
    # Row 1: Category Distribution (if available) and Year Distribution
    col1, col2 = st.columns(2)
    
    has_category = "category" in df.columns and df["category"].notna().sum() > 0
    has_year = "year" in df.columns and df["year"].notna().sum() > 0
    
    if PLOTLY_AVAILABLE:
        # Category Distribution
        with col1:
            if has_category:
                st.markdown("### Category Distribution")
                counts = df["category"].value_counts().head(15).reset_index()
                counts.columns = ["Category", "Count"]
                fig = px.bar(counts, x="Count", y="Category", orientation="h", 
                           color="Count", color_continuous_scale="Reds")
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font_color="white",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("### Source Distribution")
                if "source" in df.columns:
                    counts = df["source"].value_counts().reset_index()
                    counts.columns = ["Source", "Count"]
                    fig = px.pie(counts, values="Count", names="Source", title="Papers by Source")
                    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No category data available")
        
        # Year Distribution
        with col2:
            if has_year:
                st.markdown("### Papers by Year")
                years = df["year"].dropna().astype(int).value_counts().sort_index().reset_index()
                years.columns = ["Year", "Count"]
                fig = px.bar(years, x="Year", y="Count", color="Count", color_continuous_scale="Blues")
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font_color="white",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No year data available")
        
        # Row 2: Additional charts
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Dataset Statistics")
            stats = {
                "Total Papers": len(df),
                "With Abstract": df[next((c for c in ["original_abstract", "abstract"] if c in df.columns), df.columns[1])].notna().sum(),
                "Categories": df["category"].nunique() if has_category else "N/A",
                "Year Range": f"{int(df['year'].min())}-{int(df['year'].max())}" if has_year else "N/A",
            }
            for key, val in stats.items():
                st.metric(key, val)
        
        with col4:
            if has_category:
                st.markdown("### Top 5 Categories")
                top5 = df["category"].value_counts().head(5)
                fig = px.pie(values=top5.values, names=top5.index, hole=0.4)
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            elif has_year:
                st.markdown("### Recent Years Trend")
                recent = df[df["year"] >= df["year"].max() - 5]["year"].value_counts().sort_index()
                fig = px.line(x=recent.index, y=recent.values, markers=True, title="Last 5 Years")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly not available - showing basic charts")
        if has_category:
            st.bar_chart(df["category"].value_counts().head(10))
        if has_year:
            st.bar_chart(df["year"].value_counts().sort_index())

def viz_page():
    st.markdown("## Visualization Gallery")
    st.markdown("Pre-generated visualizations from your analysis pipeline")
    
    viz_dirs = [
        Path("/kaggle/working/visualizations"),
        Path("/kaggle/working"),
        ENV_INFO["working"] / "visualizations",
        ENV_INFO["working"],
    ]
    
    viz_files = []
    for vd in viz_dirs:
        if vd.exists():
            viz_files.extend(list(vd.glob("*.png")))
            viz_files.extend(list(vd.glob("*.jpg")))
    
    viz_files = list(set(viz_files))
    
    if viz_files:
        st.success(f"Found {len(viz_files)} visualization(s)")
        
        viz_names = {
            "All": [],
            "Categories": ["category", "distribution"],
            "Word Clouds": ["wordcloud", "word_cloud"],
            "Topics": ["topic", "bertopic", "lda"],
            "Classification": ["classification", "confusion", "lime"],
            "Dashboard": ["dashboard", "comprehensive"],
            "Embeddings": ["tsne", "embedding", "umap"],
        }
        
        tabs = st.tabs(list(viz_names.keys()))
        
        with tabs[0]:
            cols = st.columns(2)
            for i, vf in enumerate(sorted(viz_files)[:12]):
                with cols[i % 2]:
                    st.image(str(vf), caption=vf.stem, use_container_width=True)
        
        for tab_idx, (name, patterns) in enumerate(list(viz_names.items())[1:], 1):
            with tabs[tab_idx]:
                if patterns:
                    matched = [vf for vf in viz_files if any(p in vf.stem.lower() for p in patterns)]
                    if matched:
                        for vf in matched[:6]:
                            st.image(str(vf), caption=vf.stem, use_container_width=True)
                    else:
                        st.info(f"No {name.lower()} visualizations found yet")
    else:
        st.warning("No visualization files found")
        st.info("Run the visualization generation cells first, then visualizations will appear here")

if __name__ == "__main__":
    main()
