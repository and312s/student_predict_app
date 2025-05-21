import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load all model components
encoder_attend_teacher_consultancy = joblib.load("model/encoder_attend_teacher_consultancy.joblib")
encoder_ever_in_probation = joblib.load("model/encoder_ever_in_probation.joblib")
encoder_ever_suspended = joblib.load("model/encoder_ever_suspended.joblib")
encoder_meritorious_scholarship = joblib.load("model/encoder_meritorious_scholarship.joblib")
encoder_skills = joblib.load("model/encoder_skills.joblib")
encoder_relationship_status = joblib.load("model/encoder_relationship_status.joblib")

scaler_average_class_attendance = joblib.load("model/scaler_average_class_attendance.joblib")
scaler_completed_credits = joblib.load("model/scaler_completed_credits.joblib")
scaler_current_cgpa = joblib.load("model/scaler_current_cgpa.joblib")
scaler_daily_skill_dev_hours = joblib.load("model/scaler_daily_skill_dev_hours.joblib")
scaler_daily_social_media_hours = joblib.load("model/scaler_daily_social_media_hours.joblib")
scaler_daily_study_hours = joblib.load("model/scaler_daily_study_hours.joblib")
scaler_daily_study_sessions = joblib.load("model/scaler_daily_study_sessions.joblib")
scaler_monthly_family_income = joblib.load("model/scaler_monthly_family_income.joblib")
scaler_previous_sgpa = joblib.load("model/scaler_previous_sgpa.joblib")

model = joblib.load("model/rdf_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

expected_features = [
    'meritorious_scholarship', 'daily_study_hours', 'daily_study_sessions',
    'daily_social_media_hours', 'average_class_attendance', 'ever_in_probation',
    'ever_suspended', 'attend_teacher_consultancy', 'skills',
    'daily_skill_dev_hours', 'relationship_status', 'previous_sgpa',
    'current_cgpa', 'completed_credits', 'monthly_family_income'
]

# Preprocessing function
def data_preprocessing(data):
    data = data.copy()
    df = pd.DataFrame()
    df["attend_teacher_consultancy"] = encoder_attend_teacher_consultancy.transform(data["attend_teacher_consultancy"])
    df["ever_in_probation"] = encoder_ever_in_probation.transform(data["ever_in_probation"])
    df["ever_suspended"] = encoder_ever_suspended.transform(data["ever_suspended"])
    df["meritorious_scholarship"] = encoder_meritorious_scholarship.transform(data["meritorious_scholarship"])
    df["skills"] = encoder_skills.transform(data["skills"])
    df["relationship_status"] = encoder_relationship_status.transform(data["relationship_status"])
    df["average_class_attendance"] = scaler_average_class_attendance.transform(np.asarray(data["average_class_attendance"]).reshape(-1, 1))[0, 0]
    df["completed_credits"] = scaler_completed_credits.transform(np.asarray(data["completed_credits"]).reshape(-1, 1))[0, 0]
    df["current_cgpa"] = scaler_current_cgpa.transform(np.asarray(data["current_cgpa"]).reshape(-1, 1))[0, 0]
    df["daily_skill_dev_hours"] = scaler_daily_skill_dev_hours.transform(np.asarray(data["daily_skill_dev_hours"]).reshape(-1, 1))[0, 0]
    df["daily_social_media_hours"] = scaler_daily_social_media_hours.transform(np.asarray(data["daily_social_media_hours"]).reshape(-1, 1))[0, 0]
    df["daily_study_hours"] = scaler_daily_study_hours.transform(np.asarray(data["daily_study_hours"]).reshape(-1, 1))[0, 0]
    df["daily_study_sessions"] = scaler_daily_study_sessions.transform(np.asarray(data["daily_study_sessions"]).reshape(-1, 1))[0, 0]
    df["monthly_family_income"] = scaler_monthly_family_income.transform(np.asarray(data["monthly_family_income"]).reshape(-1, 1))[0, 0]
    df["previous_sgpa"] = scaler_previous_sgpa.transform(np.asarray(data["previous_sgpa"]).reshape(-1, 1))[0, 0]
    return df



# Prediction function
def prediction(data):
    data = data[expected_features]
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result


def on_predict_click(data):
    st.session_state.new_data = data_preprocessing(data)
    st.session_state.predict_triggered = True
    st.session_state.active_tab = "Penutup"


# Streamlit setup
st.set_page_config(layout="wide")

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'Beranda'


def switch_tab(tab_name):
    st.session_state.active_tab = tab_name


def go_to(page_name):
    st.session_state.active_tab = page_name


# --- Page 1: Beranda ---
if st.session_state.active_tab == 'Beranda':
    gradient_text = '''
    <style>
    .gradient-text {
        background: linear-gradient(45deg, #4E4FEB, #068FFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 36px;
        font-weight: bold;
    }
    </style>
    '''
    st.markdown(gradient_text, unsafe_allow_html=True)
    st.markdown('<p class="gradient-text"> NovaLearn: Guiding Academic Journeys Beyond the Stars</p>', unsafe_allow_html=True)
    st.caption('NovaLearn adalah aplikasi web berbasis Streamlit yang bertujuan untuk memprediksi potensi mahasiswa dropout berdasarkan data akademik dan sosial. Aplikasi ini menggunakan model machine learning untuk memberikan wawasan bagi mahasiswa terkait keberhasilan akademik mereka.')
    col1, col2 = st.columns([19.5, 1.5])
    with col2:
        st.button("Next", on_click=go_to, args=('Form',))

# --- Page 2: Form ---
if st.session_state.active_tab == 'Form':
    data = pd.DataFrame()

    # Input fields
    atc_label = "Apakah kamu pernah menghadiri konsultasi guru untuk segala jenis masalah akademis?"
    attend_teacher_consultancy = st.selectbox(label=atc_label, options=encoder_attend_teacher_consultancy.classes_, index=1)
    data["attend_teacher_consultancy"] = [attend_teacher_consultancy]

    eia_label = "Apakah kamu pernah mengikuti perbaikan nilai akademis?"
    ever_in_probation = st.selectbox(label=eia_label, options=encoder_ever_in_probation.classes_, index=1)
    data["ever_in_probation"] = [ever_in_probation]

    es_label = "Apakah kamu pernah diskors?"
    ever_suspended = st.selectbox(label=es_label, options=encoder_ever_suspended.classes_, index=1)
    data["ever_suspended"] = [ever_suspended]

    ms_label = "Apakah kamu memiliki beasiswa mahasiswa berprestasi?"
    meritorious_scholarship = st.selectbox(label=ms_label, options=encoder_meritorious_scholarship.classes_, index=1)
    data["meritorious_scholarship"] = [meritorious_scholarship]

    aca_label = "Berapa persentase rata-rata kehadiranmu di kelas?"
    average_class_attendance = int(st.number_input(label=aca_label, min_value=0, max_value=100))
    data["average_class_attendance"] = [average_class_attendance]

    skills_label = "Keterampilan apa yang kamu miliki?"
    skills = st.selectbox(label=skills_label, options=encoder_skills.classes_, index=1)
    data["skills"] = [skills]

    rs_label = 'Bagaimana status hubunganmu?'
    relationship_status = st.selectbox(label=rs_label, options=encoder_relationship_status.classes_, index=1)
    data["relationship_status"] = [relationship_status]

    dsh_label = 'Berapa jam yang kamu habiskan setiap hari untuk belajar?'
    daily_study_hours = int(st.number_input(label=dsh_label, min_value=0, max_value=24))
    data["daily_study_hours"] = [daily_study_hours]

    dss_label = 'Berapa sesi yang kamu habiskan setiap hari untuk belajar?'
    daily_study_sessions = int(st.number_input(label=dss_label, min_value=0, max_value=24))
    data["daily_study_sessions"] = [daily_study_sessions]

    dsdv_label = 'Berapa jam yang kamu habiskan setiap hari untuk pengembangan keterampilanmu?'
    daily_skill_dev_hours = int(st.number_input(label=dsdv_label, max_value=24))
    data["daily_skill_dev_hours"] = [daily_skill_dev_hours]

    dsmh_label = 'Berapa jam yang kamu habiskan setiap hari untuk sosial media?'
    daily_social_media_hours = int(st.number_input(label=dsmh_label, max_value=24))
    data["daily_social_media_hours"] = [daily_social_media_hours]

    cc_label = 'Berapa banyak SKS yang telah kamu selesaikan?'
    completed_credits = float(st.number_input(label=cc_label, max_value=100))
    data["completed_credits"] = [completed_credits]

    csgpa_label = 'Berapa IPS kamu sebelumnya?'
    previous_sgpa = float(st.number_input(label=csgpa_label, max_value=4.00))
    data["previous_sgpa"] = [previous_sgpa]

    ccgpa_label = 'Berapa IPK kamu saat ini?'
    current_cgpa = float(st.number_input(label=ccgpa_label, max_value=4.00))
    data["current_cgpa"] = [current_cgpa]

    mfi_label = 'Berapa pendapatan kamu atau keluargamu dalam sebulan?'
    monthly_family_income = float(st.number_input(label=mfi_label, min_value=0))
    data["monthly_family_income"] = [monthly_family_income]

    col1, col2, col3 = st.columns([2, 17.5, 1.5])
    with col1:
        st.button("Previous", on_click=go_to, args=('Beranda',))
    with col3:
        st.button("Predict", on_click=on_predict_click, args=(data,))

# --- Page 3: Penutup ---
if st.session_state.active_tab == 'Penutup':
    if st.session_state.get("predict_triggered", False):
        new_data = st.session_state.new_data

        hasil_prediksi = prediction(new_data)

        st.subheader("Hasil Prediksi Kelulusan Mahasiswa:")
        st.success(f"Status Mahasiswa: **{hasil_prediksi}**")
        st.session_state.predict_triggered = False
