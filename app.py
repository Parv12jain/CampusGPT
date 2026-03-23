import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralDoc · RAG Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root tokens ── */
:root {
    --bg-base:       #080c14;
    --bg-surface:    #0d1420;
    --bg-card:       #101826;
    --bg-input:      #141f2e;
    --border:        #1e2d42;
    --border-active: #2a7fff;
    --accent:        #2a7fff;
    --accent-dim:    rgba(42,127,255,.15);
    --accent-glow:   rgba(42,127,255,.35);
    --accent2:       #00e5c0;
    --accent2-dim:   rgba(0,229,192,.12);
    --text-hi:       #e8eef8;
    --text-mid:      #8a9ab5;
    --text-lo:       #3d4f65;
    --mono:          'Space Mono', monospace;
    --sans:          'Syne', sans-serif;
    --radius:        10px;
}

/* ── Reset & base ── */
html, body, [class*="css"] {
    background-color: var(--bg-base) !important;
    color: var(--text-hi) !important;
    font-family: var(--sans) !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Main area ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 960px !important;
}

/* ── Typography helpers ── */
.font-mono { font-family: var(--mono) !important; }

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0d1420 0%, #0a1628 50%, #0d1420 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent);
}
.header-banner::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(42,127,255,.07) 0%, transparent 70%);
    pointer-events: none;
}
.header-title {
    font-family: var(--sans);
    font-weight: 800;
    font-size: 1.9rem;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--text-hi) 60%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .35rem;
}
.header-sub {
    font-family: var(--mono);
    font-size: .72rem;
    color: var(--text-mid);
    letter-spacing: .08em;
    text-transform: uppercase;
}
.header-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: var(--accent-dim);
    border: 1px solid rgba(42,127,255,.3);
    color: var(--accent);
    font-family: var(--mono);
    font-size: .65rem;
    letter-spacing: .06em;
    padding: .25rem .65rem;
    border-radius: 20px;
    margin-top: .75rem;
    text-transform: uppercase;
}
.dot-live {
    width: 6px; height: 6px;
    background: var(--accent2);
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .4; transform: scale(.7); }
}

/* ── Query input box ── */
.stTextArea textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-hi) !important;
    font-family: var(--mono) !important;
    font-size: .88rem !important;
    resize: none !important;
    transition: border-color .2s !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
}
.stTextArea label { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    font-size: .85rem !important;
    letter-spacing: .04em !important;
    padding: .6rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity .18s, transform .12s !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* secondary button look via key */
[data-testid="stButton-secondary"] > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-mid) !important;
}

/* ── Answer card ── */
.answer-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    margin-top: 1.25rem;
    position: relative;
    overflow: hidden;
    animation: fade-up .35s ease;
}
.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
    border-radius: 3px 0 0 3px;
}
@keyframes fade-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.answer-label {
    font-family: var(--mono);
    font-size: .64rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .75rem;
}
.answer-text {
    font-family: var(--sans);
    font-size: .95rem;
    line-height: 1.7;
    color: var(--text-hi);
}

/* ── Source chips ── */
.sources-wrap {
    margin-top: 1.2rem;
    display: flex; flex-wrap: wrap; gap: .5rem;
}
.source-chip {
    background: var(--accent2-dim);
    border: 1px solid rgba(0,229,192,.2);
    border-radius: 6px;
    padding: .35rem .75rem;
    font-family: var(--mono);
    font-size: .68rem;
    color: var(--accent2);
    display: flex; align-items: center; gap: .4rem;
}
.source-chip .pg { color: var(--text-mid); }

/* ── Sidebar section labels ── */
.sidebar-label {
    font-family: var(--mono);
    font-size: .63rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-lo);
    margin: 1.2rem 0 .5rem;
}

/* ── Stat pill ── */
.stat-row {
    display: flex; gap: .6rem; flex-wrap: wrap;
    margin-top: 1rem;
}
.stat-pill {
    flex: 1; min-width: 90px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .65rem .8rem;
    text-align: center;
}
.stat-num {
    font-family: var(--mono);
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-desc {
    font-family: var(--mono);
    font-size: .6rem;
    color: var(--text-mid);
    letter-spacing: .05em;
    text-transform: uppercase;
}

/* ── Pipeline diagram nodes ── */
.pipeline {
    display: flex; align-items: center; gap: 0;
    margin: 1rem 0;
    overflow-x: auto;
}
.pipe-node {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .5rem .7rem;
    font-family: var(--mono);
    font-size: .62rem;
    color: var(--text-mid);
    text-align: center;
    white-space: nowrap;
    flex-shrink: 0;
}
.pipe-node.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
.pipe-arrow {
    color: var(--text-lo);
    font-size: .7rem;
    padding: 0 .25rem;
    flex-shrink: 0;
}

/* ── Chat history ── */
.chat-row {
    display: flex; gap: .75rem;
    margin-bottom: .9rem;
    animation: fade-up .25s ease;
}
.chat-avatar {
    width: 30px; height: 30px; flex-shrink: 0;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: .75rem;
    font-weight: 700;
}
.chat-avatar.user { background: var(--accent-dim); color: var(--accent); border: 1px solid rgba(42,127,255,.3); }
.chat-avatar.bot  { background: var(--accent2-dim); color: var(--accent2); border: 1px solid rgba(0,229,192,.25); }
.chat-bubble {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 0 var(--radius) var(--radius) var(--radius);
    padding: .75rem 1rem;
    font-family: var(--sans);
    font-size: .88rem;
    line-height: 1.65;
    color: var(--text-hi);
}
.chat-bubble.user {
    border-radius: var(--radius) 0 var(--radius) var(--radius);
    background: var(--bg-input);
}

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ── Streamlit misc ── */
[data-testid="stVerticalBlock"] { gap: 0 !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
[data-testid="stMarkdownContainer"] p { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data"
PERSIST_DIR = "./.chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def load_vectorstore(_embeddings):
    if not os.path.exists(PERSIST_DIR):
        loader = PyPDFDirectoryLoader(DATA_PATH)
        docs   = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=_embeddings,
            persist_directory=PERSIST_DIR,
        )
        return vs, len(chunks), len(docs)
    else:
        vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
        count = vs._collection.count()
        return vs, count, "—"

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatMistralAI(model="mistral-large-latest", temperature=0.3)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. Answer only from the provided context. "
     "If the answer is not present in the context, say: "
     "'I don't know based on the provided documents.' "
     "Keep the answer clear and concise."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"q": ..., "a": ..., "sources": [...]}


# ── Load resources ─────────────────────────────────────────────────────────────
with st.spinner("Initialising neural index…"):
    embeddings            = load_embeddings()
    vectorstore, n_chunks, n_docs = load_vectorstore(embeddings)
    llm                   = load_llm()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 1rem 1rem; border-bottom:1px solid var(--border);">
        <div style="font-family:var(--mono);font-size:.65rem;letter-spacing:.12em;
                    text-transform:uppercase;color:var(--text-lo);margin-bottom:.4rem;">
            System
        </div>
        <div style="font-family:var(--sans);font-weight:800;font-size:1.1rem;
                    color:var(--text-hi);letter-spacing:-.01em;">
            ⬡ NeuralDoc
        </div>
        <div style="font-family:var(--mono);font-size:.6rem;color:var(--text-mid);
                    margin-top:.2rem;">RAG · v1.0 · MistralAI</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown('<div class="sidebar-label">Index Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-num">{n_docs}</div>
            <div class="stat-desc">Docs</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">{n_chunks if isinstance(n_chunks,int) else "—"}</div>
            <div class="stat-desc">Chunks</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">{len(st.session_state.history)}</div>
            <div class="stat-desc">Queries</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="sidebar-label">Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
        <div class="pipe-node active">Query</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-node active">Embed</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-node active">Retrieve</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-node active">LLM</div>
    </div>
    """, unsafe_allow_html=True)

    # Config
    st.markdown('<div class="sidebar-label">Config</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:var(--mono);font-size:.67rem;color:var(--text-mid);
                line-height:1.9;background:var(--bg-card);border:1px solid var(--border);
                border-radius:var(--radius);padding:.75rem 1rem;margin-top:.25rem;">
        <span style="color:var(--text-lo);">embed  </span>MiniLM-L6-v2<br>
        <span style="color:var(--text-lo);">llm    </span>mistral-large<br>
        <span style="color:var(--text-lo);">top-k  </span>4<br>
        <span style="color:var(--text-lo);">temp   </span>0.3<br>
        <span style="color:var(--text-lo);">store  </span>ChromaDB
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    if st.button("⟳  Clear History", key="clear"):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="header-banner">
    <div class="header-title">Retrieval-Augmented Intelligence</div>
    <div class="header-sub">Query your documents with precision — powered by vector search &amp; LLM</div>
    <div class="header-badge">
        <span class="dot-live"></span>
        System online · ChromaDB indexed
    </div>
</div>
""", unsafe_allow_html=True)


# ── Chat history ───────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="sidebar-label" style="margin-bottom:.75rem;">Conversation</div>',
                unsafe_allow_html=True)
    for item in st.session_state.history:
        # user
        st.markdown(f"""
        <div class="chat-row" style="justify-content:flex-end;">
            <div class="chat-bubble user">{item["q"]}</div>
            <div class="chat-avatar user">U</div>
        </div>""", unsafe_allow_html=True)
        # assistant
        sources_html = "".join(
            f'<span class="source-chip">⊡ {s["file"]} <span class="pg">p.{s["page"]}</span></span>'
            for s in item["sources"]
        )
        st.markdown(f"""
        <div class="chat-row">
            <div class="chat-avatar bot">AI</div>
            <div>
                <div class="chat-bubble">{item["a"]}</div>
                {"<div class='sources-wrap'>" + sources_html + "</div>" if sources_html else ""}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)


# ── Input area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="sidebar-label">Ask a Question</div>', unsafe_allow_html=True)

query = st.text_area(
    label="query",
    placeholder="e.g.  What are the key findings in the uploaded documents?",
    height=100,
    key="query_input",
    label_visibility="collapsed",
)

col1, col2 = st.columns([3, 1])
with col1:
    submit = st.button("⬡  Run Query", key="submit")
with col2:
    st.markdown('<div style="height:.1rem"></div>', unsafe_allow_html=True)


# ── Query execution ────────────────────────────────────────────────────────────
if submit and query.strip():
    with st.spinner("Retrieving relevant context…"):
        retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        answer = "I don't know based on the provided documents."
        sources = []
    else:
        context = "\n\n".join(d.page_content for d in retrieved_docs)
        prompt  = PROMPT_TEMPLATE.format_messages(context=context, question=query)

        with st.spinner("Generating answer…"):
            result = llm.invoke(prompt)
        answer = result.content

        sources = [
            {
                "file": os.path.basename(d.metadata.get("source", "unknown")),
                "page": d.metadata.get("page", "?"),
            }
            for d in retrieved_docs
        ]

    # save to history
    st.session_state.history.append({"q": query, "a": answer, "sources": sources})
    st.rerun()

elif submit and not query.strip():
    st.markdown("""
    <div style="font-family:var(--mono);font-size:.72rem;color:#e8814a;
                background:rgba(232,129,74,.1);border:1px solid rgba(232,129,74,.25);
                border-radius:8px;padding:.55rem .9rem;margin-top:.5rem;">
        ⚠ Please enter a question before running the query.
    </div>
    """, unsafe_allow_html=True)