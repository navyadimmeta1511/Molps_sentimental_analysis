import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)


st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    h1, h2, h3 {
        color: #38bdf8;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0ea5e9, #4f46e5);
        color: white;
    }
    textarea {
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


st.markdown("<h1>🛍️ Product Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Analyze customer reviews instantly using Machine Learning</p>",
    unsafe_allow_html=True
)

with st.container():
    review = st.text_area(
        "✍️ Enter your product review",
        height=150,
        placeholder="Example: The product quality is amazing and delivery was fast..."
    )

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict Sentiment")

with col2:
    clear_btn = st.button("🧹 Clear Text")


if clear_btn:
    st.experimental_rerun()

if predict_btn:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before predicting")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = preprocess(review)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

        st.markdown("---")
        if prediction == 1:
            st.success("✅ **Positive Review** 😊")
        else:
            st.error("❌ **Negative Review** 😞")


st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:13px;'>
    Built with ❤️ using Streamlit & Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)