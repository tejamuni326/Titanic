import streamlit as st
import pandas as pd
import joblib
import os
import base64


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.55),
                rgba(0, 0, 0, 0.55)
            ),
            url("data:image/jpg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background("titanic.png")

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "models/titanic_best_model.pkl"

model = joblib.load(MODEL_PATH)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    font-size: 45px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    margin-bottom: 30px;
}

.info-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f0f2f6;
    margin-bottom: 20px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🚢 Titanic Survival Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Survival Prediction'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Enter the passenger information below to predict "
    "the probability of survival."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("About Project")

st.sidebar.write(
    """
    This application predicts whether a Titanic
    passenger is likely to survive based on
    passenger information.
    """
)

st.sidebar.write("### Models Used")

st.sidebar.write("""
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- ANN
- 1D CNN
""")


# ==========================================
# PASSENGER INFORMATION
# ==========================================

st.header("👤 Passenger Information")


col1, col2 = st.columns(2)


with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


with col2:

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=30.0,
        step=0.50
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )


# ==========================================
# INPUT DATA
# ==========================================

input_data = pd.DataFrame({

    "Pclass": [pclass],

    "Sex": [sex],

    "Age": [age],

    "SibSp": [sibsp],

    "Parch": [parch],

    "Fare": [fare],

    "Embarked": [embarked]

})


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.write("")

predict_button = st.button(
    "🔮 Predict Survival",
    use_container_width=True
)


if predict_button:

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0]

    survival_probability = probability[1]

    death_probability = probability[0]


    # ======================================
    # RESULT
    # ======================================

    st.header("📊 Prediction Result")


    if prediction == 1:

        st.success(
            "✅ Passenger is predicted to SURVIVE."
        )

    else:

        st.error(
            "❌ Passenger is predicted NOT TO SURVIVE."
        )


    # ======================================
    # PROBABILITIES
    # ======================================

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Survival Probability",
            f"{survival_probability:.2%}"
        )


    with result_col2:

        st.metric(
            "Non-Survival Probability",
            f"{death_probability:.2%}"
        )


    # ======================================
    # PROGRESS
    # ======================================

    st.subheader("Survival Probability")

    st.progress(
        float(survival_probability)
    )


    # ======================================
    # INPUT SUMMARY
    # ======================================

    with st.expander("View Passenger Details"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Titanic Survival Prediction | "
    "Machine Learning + Streamlit"
)