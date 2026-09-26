from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = Path(__file__).parent / "student_exam_model.joblib"
model_data = joblib.load(MODEL_PATH)

pipeline = model_data["pipeline"]
expected_columns = model_data["columns"]


st.title("Student Exam Score Prediction")
st.write("Enter the student's information to estimate the exam score.")

hours_studied = st.slider(
    "Hours Studied",
    min_value=0,
    max_value=50,
    value=10
)

attendance = st.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)

sleep_hours = st.slider(
    "Sleep Hours",
    min_value=0,
    max_value=12,
    value=7
)

previous_scores = st.slider(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=70
)

tutoring_sessions = st.slider(
    "Tutoring Sessions",
    min_value=0,
    max_value=20,
    value=2
)

physical_activity = st.slider(
    "Physical Activity",
    min_value=0,
    max_value=10,
    value=5
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

school_type = st.selectbox(
    "School Type",
    ["Public", "Private"]
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

internet_access = st.selectbox(
    "Internet Access",
    ["Yes", "No"]
)

learning_disabilities = st.selectbox(
    "Learning Disabilities",
    ["Yes", "No"]
)

peer_influence = st.selectbox(
    "Peer Influence",
    ["Positive", "Neutral", "Negative"]
)

parental_involvement = st.selectbox(
    "Parental Involvement",
    ["Low", "Medium", "High"]
)

access_to_resources = st.selectbox(
    "Access to Resources",
    ["Low", "Medium", "High"]
)

motivation_level = st.selectbox(
    "Motivation Level",
    ["Low", "Medium", "High"]
)

family_income = st.selectbox(
    "Family Income",
    ["Low", "Medium", "High"]
)

teacher_quality = st.selectbox(
    "Teacher Quality",
    ["Low", "Medium", "High"]
)

parental_education = st.selectbox(
    "Parental Education Level",
    ["High School", "College", "Postgraduate"]
)

distance_from_home = st.selectbox(
    "Distance From Home",
    ["Near", "Moderate", "Far"]
)


student = pd.DataFrame([{
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions,
    "Physical_Activity": physical_activity,

    "Gender": gender,
    "School_Type": school_type,
    "Extracurricular_Activities": extracurricular,
    "Internet_Access": internet_access,
    "Learning_Disabilities": learning_disabilities,
    "Peer_Influence": peer_influence,

    "Parental_Involvement": parental_involvement,
    "Access_to_Resources": access_to_resources,
    "Motivation_Level": motivation_level,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "Parental_Education_Level": parental_education,
    "Distance_from_Home": distance_from_home
}])

student = student[expected_columns]

if st.button("Predict Exam Score"):

    try:
        prediction = pipeline.predict(student)[0]

        st.success(f"Estimated Exam Score: {prediction:.2f}")

        if prediction < 50:
            st.warning("Student at risk")
        else:
            st.info("Student is not classified as at risk.")

    except Exception as error:
        st.error(f"Invalid input: {error}")