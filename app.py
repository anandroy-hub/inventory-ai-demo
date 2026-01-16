import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Inventory Intelligence", layout="wide", page_icon="⚙️")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 15px; border: 1px solid #e6e9ef; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("⚙️ AI-Powered Supply Chain Data Auditor")
st.markdown("""
**TPM Technical Demo:** This platform automates ETL, detects technical anomalies, and performs 'Spec-Aware' duplicate detection for industrial part inventories.
""")

# --- LOGIC & HELPERS ---
def extract_specs(text):
    """Extracts Size and Material to avoid the 'Number Trap'."""
    # Common sizes (e.g., 1/2", 10MM, 24IN)
    sizes = re.findall(r'(\d+(?:[./]\d+)?\s?(?:"|IN|MM|NB|NBX|X|NB X|INCH))', text.upper())
    # Common Industrial Materials (MOC)
    mats = re.findall(r'\b(SS316|SS304|SS316L|MS|PVC|UPVC|CPVC|GI|CS|CARBON STEEL|BRASS|TITANIUM|PP|HDPE|VITON|NEOPRENE)\b', text.upper())
    return ", ".join(set(sizes)), ", ".join(set(mats))

@st.cache_data
def run_pipeline(file_path):
    # 1. LOAD
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    # Standardize columns based on your specific 543-row format
    df.columns = ['Index', 'Item_No', 'Description', 'UoM']

    # 2. ENRICH (TPM DOMAIN LOGIC)
    # Extract Department from Prefix
    prefix_map = {
        'BM': 'CIVIL', 'CN': 'CONSUMABLES', 'IN': 'INSTRUMENTATION', 
        'IT': 'IT', 'PM': 'PRODUCTION', 'PROJ': 'PROJECTS', 
        'PS': 'STORES', 'SP': 'SPARE PARTS', 'TL': 'TOOLS'
    }
    df['Dept'] = df['Item_No'].str.extract(r'^([A-Z]+)')[0].map(prefix_map).fillna('GENERAL')
    
    # 3. CLEAN & EXTRACT SPECS
    def deep_clean(text):
        text = str(text).upper()
        text = re.sub(r'"+', '', text) # Remove quote artifacts
        return " ".join(text.split()).strip()

    df['Clean_Desc'] = df['Description'].apply(deep_clean)
    df[['Extracted_Size', 'Extracted_Material']] = df['Clean_Desc'].apply(lambda x: pd.Series(extract_specs(x)))
    
    # 4. AI CATEGORIZATION (TF-IDF + K-MEANS)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Clean_Desc'])
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(tfidf_matrix)
    
    # Auto-label clusters
    terms = tfidf.get_feature_names_out()
    cluster_labels = {}
    for i in range(8):
        top_indices = kmeans.cluster_centers_[i].argsort()[-2:][::-1]
        cluster_labels[i] = " ".join([terms[ind] for ind in top_indices]).upper()
    df['AI_SubCategory'] = df['Cluster_ID'].map(cluster_labels)

    # 5. ANOMALY DETECTION (Isolation Forest)
    df['Desc_Length'] = df['Clean_Desc'].apply(len)
    df['Complexity'] = df['Clean_Desc'].apply(lambda x: len(re.findall(r'[^A-Z0-9\s]', x)))
    iso = IsolationForest(contamination=0.03, random_state=42)
    df['Anomaly_Tag'] = iso.fit_predict(df[['Desc_Length', 'Complexity']])
    
    # 6. SMART DUPLICATE DETECTION (Fuzzy + Spec Validation)
    duplicates = []
    records = df.to_dict('records')
    for i in range(len(records)):
        for j in range(i + 1, min(i + 200, len(records))): # Optimized windowing
            r1, r2 = records[i], records[j]
            
            # Fuzzy match the text
            sim = SequenceMatcher(None, r1['Clean_Desc'], r2['Clean_Desc']).ratio()
            
            if sim > 0.85:
                # TPM Logic: If text is similar but sizes/materials differ, it's a VARIANT, not a duplicate.
                is_variant = (r1['Extracted_Size'] != r2['Extracted_Size']) or (r1['Extracted_Material'] != r2['Extracted_Material'])
                
                duplicates.append({
                    'Item A': r1['Item_No'],
                    'Item B': r2['Item_No'],
                    'Description A': r1['Clean_Desc'],
                    'Description B': r2['Clean_Desc'],
                    'Similarity': f"{sim:.1%}",
                    'Finding': "🛠️ Variant (Distinct SKU)" if is_variant else "🚨 Potential Duplicate"
                })
                
    return df, pd.DataFrame(duplicates)

# --- EXECUTION ---
try:
    df, dups = run_pipeline('raw_data.csv')
except FileNotFoundError:
    st.error("Error: 'raw_data.csv' not found. Please upload it to your GitHub repo.")
    st.stop()

# --- DASHBOARD UI ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("SKUs Analyzed", len(df))
m2.metric("Dept Categories", df['Dept'].nunique())
m3.metric("Anomalies Found", len(df[df['Anomaly_Tag'] == -1]))
m4.metric("Conflicts Found", len(dups))

tab1, tab2, tab3, tab4 = st.tabs(["📊 Inventory Structure", "🚨 Quality Audit", "👯 Duplicate Manager", "📝 Data Explorer"])

with tab1:
    st.subheader("Departmental & AI Categorization")
    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.pie(df, names='Dept', title="Inventory by Department (Prefix Logic)", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = px.bar(df['AI_SubCategory'].value_counts().head(10), title="Top AI-Detected Sub-Categories")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Data Quality Anomalies")
    st.info("Isolation Forest identified items with non-standard description patterns.")
    anomalies = df[df['Anomaly_Tag'] == -1]
    st.dataframe(anomalies[['Item_No', 'Description', 'Dept', 'AI_SubCategory']], use_container_width=True)
    
    fig3 = px.scatter(df, x="Desc_Length", y="Complexity", color="Anomaly_Tag", 
                     hover_data=['Clean_Desc'], title="Anomaly Distribution Graph")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Smart Duplicate Management")
    st.warning("Logic: Items with similar text but different technical specs (Size/Material) are marked as 'Variants'.")
    
    if not dups.empty:
        # Filter for only true duplicates or variants
        view_filter = st.radio("Filter Finding:", ["All Conflicts", "Potential Duplicates", "Variants"])
        if view_filter != "All Conflicts":
            display_dups = dups[dups['Finding'].str.contains(view_filter.split()[0])]
        else:
            display_dups = dups
        st.dataframe(display_dups, use_container_width=True)
    else:
        st.success("No conflicts detected.")

with tab4:
    st.subheader("Full Processed Dataset")
    st.dataframe(df[['Item_No', 'Clean_Desc', 'UoM', 'Dept', 'AI_SubCategory', 'Extracted_Size', 'Extracted_Material']])
