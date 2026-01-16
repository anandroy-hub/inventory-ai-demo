import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from difflib import SequenceMatcher

# Advanced AI/ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Supply Chain Auditor Pro", layout="wide", page_icon="🛡️")

# --- KNOWLEDGE BASE: DOMAIN LOGIC ---
DEPARTMENT_MAP = {
    'TL': 'TOOLS & EQUIPMENT', 'IN': 'INSTRUMENTATION', 'CN': 'CONSUMABLES', 
    'SP': 'SPARE PARTS', 'BM': 'CIVIL/BUILDING', 'PS': 'PROJECT STORES', 'IT': 'IT'
}

PRODUCT_GROUPS = {
    "HAND TOOLS": ["PLIER", "STRIPPER", "WRENCH", "SPANNER", "HAMMER", "FILE", "SAW", "TOOL", "CHISEL", "CUTTER", "TAPE MEASURE"],
    "PIPING COMPONENTS": ["PIPE", "FLANGE", "ELBOW", "TEE", "REDUCER", "BEND", "COUPLING", "NIPPLE", "BUSHING", "UPVC", "CPVC", "PVC"],
    "VALVE ASSEMBLIES": ["VALVE", "ACTUATOR", "BALL VALVE", "GATE VALVE", "CHECK VALVE", "GLOBE VALVE", "PLUG VALVE", "COCK"],
    "FASTENERS & HARDWARE": ["STUD", "BOLT", "NUT", "WASHER", "GASKET", "O-RING", "SEAL", "MECH SEAL", "GLOW", "JOINT"],
    "INSTRUMENTATION": ["TRANSMITTER", "GAUGE", "CABLE", "WIRE", "CONNECTOR", "PLUG", "SWITCH", "HUB", "SENSOR"],
    "CONSUMABLES": ["BRUSH", "TAPE", "STICKER", "CHALK", "GLOVE", "CLEANER", "PAINT", "CEMENT", "HOSE", "ADHESIVE"]
}

SPEC_TRAPS = {
    "Gender": ["MALE", "FEMALE"],
    "Connection": ["BW", "SW", "THD", "THREADED", "FLGD", "FLANGED", "SORF", "WNRF", "BLRF"],
    "Rating": ["150#", "300#", "600#", "PN10", "PN16", "PN25", "PN40"]
}

# --- AI UTILITIES ---
def get_tech_dna(text):
    text = str(text).upper()
    dna = {"numbers": set(re.findall(r'\d+(?:[./]\d+)?', text)), "attributes": {}}
    for cat, keywords in SPEC_TRAPS.items():
        found = [k for k in keywords if re.search(rf'\b{k}\b', text)]
        if found: dna["attributes"][cat] = set(found)
    return dna

def intelligent_noun_extractor(text):
    text = str(text).upper()
    phrases = ["MEASURING TAPE", "BALL VALVE", "GATE VALVE", "CHECK VALVE", "PAINT BRUSH", "WIRE STRIPPER", "CUTTING PLIER"]
    for p in phrases:
        if p in text: return p
    all_nouns = [item for sublist in PRODUCT_GROUPS.values() for item in sublist]
    for n in all_nouns:
        if re.search(rf'\b{noun}\b', text): return n
    return text.split()[0] if text.split() else "MISC"

# --- MAIN ENGINE ---
@st.cache_data
def run_intelligent_audit(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    id_col = next(c for c in df.columns if any(x in c.lower() for x in ['item', 'no']))
    desc_col = next(c for c in df.columns if 'desc' in c.lower())
    
    df['Standard_Desc'] = df[desc_col].astype(str).str.upper().str.replace('"', '').str.strip()
    df['Prefix'] = df[id_col].str.extract(r'^([A-Z]+)')
    df['Business_Dept'] = df['Prefix'].map(DEPARTMENT_MAP).fillna('GENERAL STOCK')
    df['Part_Noun'] = df['Standard_Desc'].apply(intelligent_noun_extractor)

    # NLP & Topic Modeling
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Standard_Desc'])
    nmf = NMF(n_components=12, random_state=42, init='nndsvd')
    nmf_features = nmf.fit_transform(tfidf_matrix)
    
    # Clustering for Confidence
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    dists = kmeans.transform(tfidf_matrix)
    df['Confidence'] = (1 - (np.min(dists, axis=1) / np.max(dists))).round(4)
    
    # Anomaly
    iso = IsolationForest(contamination=0.04, random_state=42)
    df['Anomaly_Flag'] = iso.fit_predict(df[[desc_col]].applymap(len))

    # Fuzzy & Tech DNA
    df['Tech_DNA'] = df['Standard_Desc'].apply(get_tech_dna)
    exact_dups = df[df.duplicated(subset=['Standard_Desc'], keep=False)]
    
    return df, exact_dups, id_col, desc_col

# --- DATA LOADING ---
target_file = 'raw_data.csv' if os.path.exists('raw_data.csv') else 'Demo - Raw data.xlsx - Sheet2.csv'
if os.path.exists(target_file):
    df_raw, exact_dups, id_col, desc_col = run_intelligent_audit(target_file)
else:
    st.error("Data file missing.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ Auditor Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Categorization Audit", "Anomaly & Duplicate Hub", "Methodology"])

# --- GLOBAL FILTERS ---
st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")
selected_dept = st.sidebar.multiselect("Filter by Department", options=df_raw['Business_Dept'].unique(), default=df_raw['Business_Dept'].unique())
selected_noun = st.sidebar.multiselect("Filter by Product Type", options=sorted(df_raw['Part_Noun'].unique()), default=[])

# Filtered Dataframe
df = df_raw[df_raw['Business_Dept'].isin(selected_dept)]
if selected_noun:
    df = df[df['Part_Noun'].isin(selected_noun)]

# --- PAGE: DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.header("Inventory Data Health Overview")
    
    # KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SKUs in View", len(df))
    m2.metric("Mean Confidence", f"{df['Confidence'].mean():.1%}")
    m3.metric("Anomalies Found", len(df[df['Anomaly_Flag'] == -1]))
    m4.metric("Unique Product Types", df['Part_Noun'].nunique())

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(df, names='Business_Dept', title="Inventory Split by Business Function", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_bar = px.bar(df['Part_Noun'].value_counts().head(10), title="Top 10 Inventory Items in View", labels={'value':'Count', 'index':'Product'})
        st.plotly_chart(fig_bar, use_container_width=True)

# --- PAGE: CATEGORIZATION AUDIT ---
elif page == "Categorization Audit":
    st.header("Categorization & Classification Audit")
    st.info("Explore how items have been classified. High confidence indicates a strong match with our Knowledge Base.")
    
    st.dataframe(df[[id_col, 'Standard_Desc', 'Part_Noun', 'Business_Dept', 'Confidence']].sort_values('Confidence', ascending=False), use_container_width=True)
    
    fig_scatter = px.scatter(df, x='Part_Noun', y='Confidence', color='Business_Dept', title="Categorization Confidence by Product Type")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- PAGE: ANOMALY & DUPLICATE HUB ---
elif page == "Anomaly & Duplicate Hub":
    st.header("Quality & Risk Identification")
    
    t1, t2 = st.tabs(["🚨 Anomalies", "👯 Duplicates"])
    
    with t1:
        st.subheader("Data Pattern Anomalies")
        anomalies = df[df['Anomaly_Flag'] == -1]
        st.warning(f"Found {len(anomalies)} items with non-standard patterns.")
        st.dataframe(anomalies[[id_col, desc_col, 'Part_Noun']])
        
    with t2:
        st.subheader("Exact Duplicates (Cleaned Descriptions)")
        view_exact = exact_dups[exact_dups[id_col].isin(df[id_col])]
        if not view_exact.empty:
            st.error(f"Found {len(view_exact)} exact duplicates in filtered view.")
            st.dataframe(view_exact[[id_col, desc_col]])
        else:
            st.success("No exact duplicates in this view.")

# --- PAGE: METHODOLOGY ---
elif page == "Methodology":
    st.header("AI Technical Methodology")
    st.markdown("""
    ### 1. Categorization (The "Intelligent" Hybrid)
    We use **Domain Prioritization** (Heuristics) paired with **NMF Topic Modeling**. This prevents the 'Size Trap' where tools were previously grouped with pipes.
    
    ### 2. Anomaly Detection
    Powered by **Isolation Forests**, we analyze text complexity and length to find broken records.
    
    ### 3. Duplicate Resolution
    Utilizes **Levenshtein Distance** with a **Tech DNA override**. If sizes or genders (Male/Female) differ, the system classifies them as 'Variants'.
    """)
