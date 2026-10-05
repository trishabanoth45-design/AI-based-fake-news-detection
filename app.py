import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import pipeline
import spacy
from web_scraper import extract_article_data
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)
# Cache the NLP model loading to avoid reloading on every interaction
@st.cache_resource
def load_models():
    # Load Hugging Face Transformer Pipeline for Sequence Classification
    # Note: Replace with your custom fine-tuned fake news model if available
    hf_classifier = pipeline(
        "text-classification", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Load Spacy for entity & text statistics
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback if model isn't linked
        nlp = None
        
    return hf_classifier, nlp

classifier, nlp = load_models()

# App Header
st.title("📰 AI-Driven Fake News Detection Platform")
st.markdown("Analyze news articles using Deep Learning and Natural Language Processing to determine integrity metrics.")

# Sidebar Controls
st.sidebar.header("🔧 Configuration")
analysis_mode = st.sidebar.radio("Choose Input Type:", ["Direct Text Input", "Fetch via Article URL"])
confidence_threshold = st.sidebar.slider("Confidence Alert Threshold", 0.5, 1.0, 0.75)

# Main UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Article Analysis Panel")
    
    article_title = ""
    article_body = ""
    domain_info = "N/A"
    
    if analysis_mode == "Fetch via Article URL":
        url_input = st.text_input("Paste News Article URL:")
        if st.button("Fetch and Extract Content"):
            if url_input:
                with st.spinner("Scraping article contents securely..."):
                    result = extract_article_data(url_input)
                    if result["success"]:
                        article_title = result["title"]
                        article_body = result["content"]
                        domain_info = result["domain"]
                        st.success(f"Successfully imported from {domain_info}!")
                    else:
                        st.error(f"Failed to fetch webpage: {result['error']}")
            else:
                st.warning("Please provide a valid URL.")
                
        # Editable preview panels for scraped content
        article_title = st.text_input("Extracted Title:", value=article_title)
        article_body = st.text_area("Extracted Body Text:", value=article_body, height=300)
        
    else:
        article_title = st.text_input("Enter Article Headline:")
        article_body = st.text_area("Enter Full Article Body Text:", height=300)

    # Trigger Model Inference
    if st.button("🚀 Run AI Credibility Assessment"):
        if not article_body.strip():
            st.warning("Please enter text content to analyze.")
        else:
            with st.spinner("Processing deep structural and semantic analysis..."):
                # 1. Transformers Inference
                # Combining title and body for holistic semantic analysis
                full_text = f"{article_title}. {article_body}"[:2000] 
                prediction = classifier(full_text)[0]
                
                # Mapping placeholder fine-tuned outputs to Fake/Real terminology
                # (Assuming POSITIVE=REAL, NEGATIVE=FAKE for simulation if using default pipeline)
                is_fake = prediction['label'] == 'NEGATIVE' 
                score = prediction['score']
                
                # 2. Extract Basic Visual Analytics Metrics
                word_count = len(article_body.split())
                char_count = len(article_body)
                
                # Display Results Card
                st.markdown("---")
                st.subheader("📊 Evaluation Report")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    if is_fake:
                        st.metric(label="Credibility Score", value="🔴 HIGH RISK FAKE", delta=f"{score:.2%} Match")
                    else:
                        st.metric(label="Credibility Score", value="🟢 VERIFIED / REAL", delta=f"{score:.2%} Match")
                        
                with metric_col2:
                    st.metric(label="Source Domain", value=domain_info)
                with metric_col3:
                    st.metric(label="Word Metric Count", value=f"{word_count} words")

                # Data Visualization Column integration
                with col2:
                    st.subheader("📈 Linguistic Signatures")
                    
                    # Sentiment / Probability Matrix Chart
                    fig, ax = plt.subplots(figsize=(4, 3))
                    labels = ['Fake Bias Probability', 'Real Bias Probability']
                    values = [score, 1 - score] if is_fake else [1 - score, score]
                    
                    sns.barplot(x=labels, y=values, palette="RdYlGn_r" if is_fake else "RdYlGn", ax=ax)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Confidence Level')
                    st.pyplot(fig)
                    
                    # Entity Extraction Statistics via SpaCy
                    if nlp and article_body:
                        doc = nlp(article_body[:1000])
                        entities = [ent.label_ for ent in doc.ents]
                        if entities:
                            st.write("**Identified Entity Densities:**")
                            ent_df = pd.DataFrame(entities, columns=['Entity Type']).value_counts().reset_index()
                            st.dataframe(ent_df, use_container_width=True)
                        else:
                            st.caption("No dense Named Entities found in sample text segment.")

# Baseline Overview Metrics when app is idle
if not article_body.strip():
    with col2:
        st.subheader("💡 Dashboard Insights")
        st.info("Provide text input or a web link on the left panel to execute real-time fake news tracking algorithms.")