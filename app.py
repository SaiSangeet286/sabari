import streamlit as st

st.set_page_config(page_title="Customer Feedback Insight Generator", layout="wide")

import pandas as pd
import nltk
from rake_nltk import Rake
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


# Load the sentiment analysis model from the local "saved_model" directory
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("saved_model")
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)


sentiment_pipeline = load_model()
labels = ["negative", "neutral", "positive"]


# Function to generate recommendations based on sentiment and themes
def get_recommendations(sentiment, themes):
    sentiment = sentiment.lower()
    if sentiment == 'negative':
        return f"🔍 Investigate issues related to: {', '.join(themes)}"
    elif sentiment == 'positive':
        return f"✅ Continue doing well in areas like: {', '.join(themes)}"
    else:
        return f"📌 Monitor themes like: {', '.join(themes)} for future improvement."


# Analyze a single feedback entry
def analyze_feedback(feedback_text):
    result = sentiment_pipeline(feedback_text[:512])[0]
    scores = {labels[i]: result[i]["score"] for i in range(len(result))}
    sentiment = max(scores, key=scores.get)

    r = Rake()
    r.extract_keywords_from_text(feedback_text)
    themes = r.get_ranked_phrases()[:3]

    recommendation = get_recommendations(sentiment, themes)
    return sentiment.capitalize(), themes, recommendation


# UI
st.title("📝 Customer Feedback Insight Generator (Offline Model)")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'feedback' not in df.columns:
        st.error("❌ The CSV must contain a 'feedback' column.")
    else:
        st.success("✅ CSV uploaded successfully.")
        feedback_data = []

        with st.spinner("🔍 Analyzing feedback..."):
            for feedback in df['feedback']:
                sentiment, themes, recommendation = analyze_feedback(feedback)
                feedback_data.append({
                    'Feedback': feedback,
                    'Sentiment': sentiment,
                    'Themes': ", ".join(themes),
                    'Recommendations': recommendation
                })

        result_df = pd.DataFrame(feedback_data)

        st.header("📊 Sentiment Analysis Results")
        st.dataframe(result_df[['Feedback', 'Sentiment']])

        st.header("🔍 Common Themes")
        st.dataframe(result_df[['Feedback', 'Themes']])

        st.header("✅ Recommendations")
        for _, row in result_df.iterrows():
            st.markdown(f"**🗣️ Feedback:** {row['Feedback']}")
            st.markdown(f"**💡 Recommendation:** {row['Recommendations']}")
            st.markdown("---")
else:
    st.info("📁 Please upload a CSV file with a 'feedback' column.")