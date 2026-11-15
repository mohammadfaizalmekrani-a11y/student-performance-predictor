
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------
# LOAD AI MODEL
# -------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "student_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load AI model: {e}")
    st.stop()

# -------------------------------
# WEBSITE DESIGN
# -------------------------------

st.title("🎓 Student Performance Predictor")
st.write(
    "An AI-powered application to predict student academic performance."
)

st.divider()

st.subheader("📚 Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Daily Study Hours",
        0.0, 10.0, 5.0, 0.5
    )

    attendance = st.slider(
        "Attendance (%)",
        40, 100, 85
    )

with col2:
    st.info("Enter marks for each subject out of 50.")

# -------------------------------
# SUBJECT MARKS
# -------------------------------

st.subheader("📝 Subject-wise Marks")

c1, c2, c3 = st.columns(3)

with c1:
    physics = st.number_input(
        "Physics", 0, 50, 35
    )
    mathematics = st.number_input(
        "Mathematics", 0, 50, 35
    )

with c2:
    c_programming = st.number_input(
        "C Programming", 0, 50, 35
    )
    ai_marks = st.number_input(
        "Artificial Intelligence", 0, 50, 35
    )

with c3:
    electronics = st.number_input(
        "Electronics", 0, 50, 35
    )
    assignment_marks = st.number_input(
        "Assignment", 0, 50, 35
    )

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("🚀 Predict Student Performance", use_container_width=True):

    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "physics": physics,
        "mathematics": mathematics,
        "c_programming": c_programming,
        "ai_marks": ai_marks,
        "electronics": electronics,
        "assignment_marks": assignment_marks
    }])

    try:
        with st.spinner("AI is predicting performance..."):
            prediction = float(model.predict(input_data)[0])

        prediction = max(0, min(50, prediction))

        average = (
            physics + mathematics + c_programming +
            ai_marks + electronics + assignment_marks
        ) / 6

        if prediction >= 45:
            category = "Excellent"
        elif prediction >= 30:
            category = "Good"
        elif prediction >= 20:
            category = "Average"
        else:
            category = "Needs Improvement"

        st.divider()
        st.subheader("📊 Prediction Results")

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Predicted Performance",
                f"{prediction:.2f} / 50"
            )

        with r2:
            st.metric(
                "Subject Average",
                f"{average:.2f} / 50"
            )

        st.success(f"Performance Category: {category}")

        st.progress(prediction / 50)

        st.caption(
            "This is an estimated result from a machine-learning "
            "model, not a guaranteed examination score."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

st.caption("Student Performance Predictor | AI & ML Mini Project")
