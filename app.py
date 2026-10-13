
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

MODEL_PATH = Path("fake_news_model.joblib")

st.title("📰 AI Fake News Detector")
st.write(
    "Enter a news headline or article to get a machine-learning prediction."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

news_text = st.text_area(
    "Paste news content here",
    height=200,
    placeholder="Enter a news headline or article..."
)

if st.button("Analyze News", type="primary"):
    if not news_text.strip():
        st.warning("Please enter some news text.")
    else:
        prediction = model.predict([news_text])[0]

        st.subheader("Prediction")

        if prediction == "FAKE":
            st.error("⚠️ AI prediction: FAKE NEWS")
        else:
            st.success("✅ AI prediction: REAL NEWS")

        probabilities = model.predict_proba([news_text])[0]
        classes = model.classes_

        st.subheader("Model scores")
        for label, score in zip(classes, probabilities):
            st.write(f"{label}: {score * 100:.2f}%")

        st.info(
            "These scores reflect the model's learned classification, "
            "not the verified truth of the article. Check reliable "
            "independent sources before sharing the news."
        )
