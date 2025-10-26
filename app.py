import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("lr_model.jb")
vectorizer = joblib.load("vectorizer.jb")

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Session state for input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background: linear-gradient(to right, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1) !important;
            font-family: sans-serif;
            color: red;
        }
        
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #2c3e50;
        }

        .subtitle {
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
            color: #34495e;
        }

        .stTextArea textarea {
            background-color: #ffffff;
            color: #2c3e50;
            border: 2px solid #a18cd1;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            box-shadow: inset 0 0 8px rgba(161, 140, 209, 0.3);
        }

        .stButton>button {
            background: linear-gradient(to right, #a18cd1, #fbc2eb);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 2em;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(161, 140, 209, 0.3);
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background: linear-gradient(to right, #fbc2eb, #a18cd1);
            box-shadow: 0 6px 20px rgba(161, 140, 209, 0.5);
        }

        .result-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 25px;
            margin-top: 25px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .real {
            color: #27ae60;
            font-weight: bold;
            font-size: 24px;
        }

        .fake {
            color: #e74c3c;
            font-weight: bold;
            font-size: 24px;
        }

        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
        }

        .footer-bar {
            height: 6px;
            background: linear-gradient(to right, #a18cd1, #fbc2eb);
            border-radius: 3px;
            margin-top: 20px;
        }
    </style>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">📰 FAKE NEWS DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a news headline or article snippet to check its authenticity.</div>', unsafe_allow_html=True)

# Input
st.session_state.user_input = st.text_area("📝 Enter News Heading", value=st.session_state.user_input, height=150)

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔍 Analyze"):
        if st.session_state.user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            vec = vectorizer.transform([st.session_state.user_input])
            prediction = model.predict(vec)[0]

            result_class = "real" if prediction == 1 else "fake"
            result_text = "✅ This news appears to be REAL." if prediction == 1 else "🚫 This news appears to be FAKE."

            st.markdown(f"""
                <div class="result-card">
                    <div class="{result_class}">{result_text}</div>
                </div>
            """, unsafe_allow_html=True)

with col2:
    if st.button("🧹 Clear"):
        st.session_state.user_input = ""

# Footer
st.markdown('<div class="footer">Made with ❤ using Streamlit</div>', unsafe_allow_html=True)
st.markdown('<div class="footer-bar"></div>', unsafe_allow_html=True)