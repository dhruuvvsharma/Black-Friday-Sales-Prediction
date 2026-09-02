import re
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
PRODUCT_ID_PATTERN = re.compile(r"^P\d{8}$")

st.set_page_config(
    page_title="Black Friday Sales Prediction",
    page_icon="🛍️",
    layout="centered",
)

# Custom CSS for blue header and smooth light background
st.markdown("""
<style>
/* Hide default Streamlit elements */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* App Background */
.stApp {
    background-color: #f8f9fa !important;
}

/* Custom blue header */
.custom-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 0 0 25px 25px;
    margin: -3.5rem -1rem 2rem -1rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}
.header-title {
    color: #FFD700 !important;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.25);
}
.header-subtitle {
    color: #FFFFFF !important;
    font-size: 1rem;
    margin-top: 0.5rem;
    opacity: 0.95;
}

/* --- FIXING THE LIGHT THEME --- */

/* Force dark text for all labels, headings, and standard text */
label, h2, h3, .stMarkdown p, .stAlert {
    color: #262730 !important;
}

/* Force white background and dark text for standard inputs */
input, select, textarea {
    background-color: #ffffff !important;
    color: #262730 !important;
    border: 1px solid #d1d5db !important;
}

/* Fix the dark +/- buttons in Number Input */
.stNumberInput button {
    background-color: #ffffff !important;
    color: #262730 !important;
    border: 1px solid #d1d5db !important;
}

/* Fix Selectbox dropdown containers */
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
}
.stSelectbox div {
    color: #262730 !important;
}

/* Sidebar styling */
.stSidebar {
    background-color: #ffffff !important;
}
.stSidebar label, .stSidebar p, .stSidebar h3 {
    color: #262730 !important;
}

/* Success card */
.success-card {
    padding: 1.5rem;
    border-radius: 0.75rem;
    background-color: #f0f7f0;
    border: 1px solid #c6e6c6;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Buttons */
.stButton > button {
    background-color: #4169E1 !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
}
.stButton > button:hover {
    background-color: #3150b8 !important;
    box-shadow: 0 4px 12px rgba(65, 105, 225, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# Custom Header
st.markdown("""
<div class="custom-header">
    <h1 class="header-title">🛍️ Black Friday Sales Prediction</h1>
    <p class="header-subtitle">Powered by Advanced ML</p>
</div>
""", unsafe_allow_html=True)

st.write("Enter customer and product details to predict the purchase amount.")

# Quick backend status check
with st.sidebar:
    st.subheader("API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("Backend is online")
        else:
            st.error("Backend reports unhealthy")
    except requests.exceptions.ConnectionError:
        st.error("Backend unreachable")

st.divider()

st.subheader("Customer Information")
col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("User ID", min_value=1000000, max_value=1010000, value=1000001, step=1)
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.selectbox("Age", ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"])
with col2:
    occupation = st.number_input("Occupation", min_value=0, max_value=20, value=10, step=1)
    city_category = st.selectbox("City Category", ["A", "B", "C"])
    stay_in_city = st.selectbox("Stay In Current City (Years)", ["0", "1", "2", "3", "4+"])

marital_status = st.radio("Marital Status", [0, 1], format_func=lambda x: "Married" if x == 1 else "Single", horizontal=True)

st.divider()
st.subheader("Product Information")

product_id = st.text_input("Product ID", value="P00069042", help="Format: P followed by 8 digits, e.g. P00069042")

col3, col4, col5 = st.columns(3)
with col3:
    product_category_1 = st.number_input("Category 1", min_value=1, max_value=20, value=3, step=1)
with col4:
    product_category_2 = st.selectbox(
        "Category 2 (optional)",
        options=[None] + list(range(1, 21)),
        index=0,
    )
with col5:
    product_category_3 = st.selectbox(
        "Category 3 (optional)",
        options=[None] + list(range(1, 21)),
        index=0,
        disabled=(product_category_2 is None),
        help="Requires Category 2 to be set first" if product_category_2 is None else None,
    )

st.divider()
submitted = st.button("Predict Purchase", type="primary", use_container_width=True)

if submitted:
    errors = []
    if not PRODUCT_ID_PATTERN.match(product_id):
        errors.append("Product ID must be in the format P followed by 8 digits (e.g. P00069042).")
    if product_category_3 is not None and product_category_2 is None:
        errors.append("Category 3 requires Category 2 to be set.")

    if errors:
        for err in errors:
            st.warning(err)
    else:
        input_data = {
            "User_ID": user_id,
            "Product_ID": product_id,
            "Gender": gender,
            "Age": age,
            "Occupation": occupation,
            "City_Category": city_category,
            "Stay_In_Current_City_Years": stay_in_city,
            "Marital_Status": marital_status,
            "Product_Category_1": product_category_1,
            "Product_Category_2": product_category_2,
            "Product_Category_3": product_category_3,
        }

        try:
            with st.spinner("Running prediction..."):
                response = requests.post(f"{API_URL}/predict", json=input_data, timeout=10)

            if response.status_code == 200:
                prediction = response.json()["predicted_purchase"]
                st.divider()
                st.markdown(
                    f"""
                    <div class="success-card">
                        <p style="margin: 0; font-size: 0.9rem; color: #4a4a4a;">Predicted Purchase Amount</p>
                        <p style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #1b6b1b;">
                            ₹{prediction:,.2f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif response.status_code == 422:
                detail = response.json().get("detail", "Invalid input")
                if isinstance(detail, list):
                    for item in detail:
                        st.error(item.get("msg", str(item)))
                else:
                    st.error(detail)

            elif response.status_code == 503:
                st.error("Model is currently unavailable on the server. Please try again shortly.")

            else:
                st.error(f"API error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")
        except requests.exceptions.Timeout:
            st.error("The request timed out. The server may be under load — try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"Unexpected request error: {e}")