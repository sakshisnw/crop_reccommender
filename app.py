import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="Crop Recommender", page_icon="🌱", layout="centered")

# Custom CSS for full dark text and sage green background
st.markdown(
    """
    <style>
    /* Top header toolbar fix */
    header[data-testid="stHeader"] {
        background-color: #dfe9d1 !important;
        color: #1a1a1a !important;
    }

    /* Force all toolbar icons/text visible */
    header[data-testid="stHeader"] * {
        color: #1a1a1a !important;
    }
    </style>

    
    <style>
    /* Entire background */
    html, body, .stApp {
        background-color: #dfe9d1 !important;
        color: #1a1a1a !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #e4f0d0 !important;
        color: #1a1a1a !important;
    }

    /* Input boxes (number input) */
    input[type="number"] {
        background-color: #f4f4f4 !important;
        color: #1a1a1a !important;
        border: 1px solid #aaa !important;
        border-radius: 6px;
        padding: 6px;
    }

    /* Input labels */
    label, .css-1aumxhk, .stNumberInput label {
        color: #1a1a1a !important;
        font-weight: 600;
    }

    /* Button */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5em 1em;
    }

    /* Dataframe text color */
    .stDataFrame th, .stDataFrame td {
        color: #1a1a1a !important;
    }

    /* Title and paragraphs */
    h1, h2, h3, p, .stMarkdown {
        color: #1a1a1a !important;
    }

    hr {
        border-top: 1px solid #aaa;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Load model, scaler, and label encoder
@st.cache_resource
def load_artifacts():
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()

# Features
feature_names = ['Ph', 'K', 'P', 'N']

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #1a1a1a;'> Crop Recommendation System </h1>
    <p style='text-align: center; color: #1a1a1a;'>Enter your soil parameters below to find the most suitable crop.</p>
    """,
    unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/7667/7667876.png", width=100)
st.sidebar.header("Input Soil Nutrient Values")

user_input = {}
for feature in feature_names:
    default = 6.0 if feature == "Ph" else 100.0
    user_input[feature] = st.sidebar.number_input(
        label=f"Enter {feature} value",
        min_value=0.0,
        max_value=300.0 if feature != "Ph" else 14.0,
        value=default,
        step=0.1,
        format="%.2f"
    )

input_df = pd.DataFrame([user_input])

# Main content
st.markdown("---")
st.subheader(" Entered Soil Data")
st.dataframe(input_df, use_container_width=True)

# Prediction
if st.button("Recommend Best Crop"):
    scaled_input = scaler.transform(input_df)
    pred_encoded = model.predict(scaled_input)
    pred_label = label_encoder.inverse_transform(pred_encoded)[0]
    
    st.success(f"Recommended Crop: **{pred_label}**")
    st.balloons()

