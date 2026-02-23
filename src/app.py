import streamlit as st
import pandas as pd
import joblib
import os

# 1. ตั้งค่าหน้าเว็บให้เต็มจอ (Wide) เพื่อรับมือกับ 11 ตัวแปร
st.set_page_config(page_title="Exoplanet Classifier", page_icon="🔭", layout="wide")

# 2. แอบใส่ Custom CSS ให้ดูทันสมัยขึ้น (ปรับขนาดฟอนต์และสี)
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #1E90FF; }
    </style>
    """, unsafe_allow_html=True)

# 3. Header
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🔭 Deep-Space Exoplanet Validator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #808495;'>Advanced 11-Dimensional Telemetry Analysis Powered by XGBoost.</p>", unsafe_allow_html=True)
st.markdown("---")

model_path = 'models/kepler_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    # 4. ส่วนเลือกข้อมูล (ไว้ตรงกลางให้เด่นๆ)
    col_preset1, col_preset2, col_preset3 = st.columns([1, 2, 1])
    with col_preset2:
        preset_options = {
            "Manual Input (กรอกเอง)": [0.0]*11,
            "Kepler-22b (Confirmed Habitable)": [289.86, 7.40, 492.0, 2.38, 35.8, 262.0, 0.98, 5518.0, 4.44, 1.11, 15.3],
            "Kepler-186f (Confirmed Earth-size)": [129.94, 5.28, 124.0, 1.11, 24.3, 188.0, 0.47, 3788.0, 4.81, 0.29, 14.6],
            "Typical False Positive (Binary Star)": [1.50, 2.00, 50000.0, 15.00, 1500.0, 4500.0, 1.20, 6000.0, 4.00, 10000.0, 10.5]
        }
        selected_preset = st.selectbox("📂 Load Presets (โหลดข้อมูลตัวอย่าง):", list(preset_options.keys()))
        default_vals = preset_options[selected_preset]

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. แผงควบคุม 11 ช่อง (Card Design แบบ 3 แถว)
    st.markdown("### 🎛️ Telemetry Parameters")
    
    # แถวที่ 1: การเคลื่อนที่
    with st.container(border=True):
        st.markdown("**1. Transit Signatures (ข้อมูลการเคลื่อนที่ผ่านหน้าดาวแม่)**")
        col1, col2, col3, col4 = st.columns(4)
        with col1: koi_period = st.number_input("Period (Days)", value=default_vals[0])
        with col2: koi_duration = st.number_input("Duration (Hours)", value=default_vals[1])
        with col3: koi_depth = st.number_input("Depth (PPM)", value=default_vals[2])
        with col4: koi_prad = st.number_input("Planet Radius (Earth=1)", value=default_vals[3])

    # แถวที่ 2: ดาวแม่และอุณหภูมิ
    with st.container(border=True):
        st.markdown("**2. Stellar & Environmental Properties (ข้อมูลดาวฤกษ์และอุณหภูมิ)**")
        col5, col6, col7, col8 = st.columns(4)
        with col5: koi_model_snr = st.number_input("Signal-to-Noise", value=default_vals[4])
        with col6: koi_teq = st.number_input("Equilibrium Temp (K)", value=default_vals[5])
        with col7: koi_srad = st.number_input("Stellar Radius (Sun=1)", value=default_vals[6])
        with col8: koi_steff = st.number_input("Stellar Temp (K)", value=default_vals[7])

    # แถวที่ 3: แรงโน้มถ่วงเชิงลึก
    with st.container(border=True):
        st.markdown("**3. Advanced Gravity & Optics (ข้อมูลเชิงลึกทางฟิสิกส์)**")
        col9, col10, col11 = st.columns(3)
        with col9: koi_slogg = st.number_input("Gravity (log(g))", value=default_vals[8])
        with col10: koi_insol = st.number_input("Insolation Flux", value=default_vals[9])
        with col11: koi_kepmag = st.number_input("Kepler Magnitude", value=default_vals[10])

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. ปุ่มกดและผลลัพธ์
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🚀 Execute Ultra-Classification Protocol", use_container_width=True, type="primary"):
            input_data = pd.DataFrame([[koi_period, koi_duration, koi_depth, koi_prad, koi_model_snr, koi_teq, koi_srad, koi_steff, koi_slogg, koi_insol, koi_kepmag]], 
                                       columns=['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_model_snr', 'koi_teq', 'koi_srad', 'koi_steff', 'koi_slogg', 'koi_insol', 'koi_kepmag'])
            
            with st.spinner('Analyzing 11-dimensional telemetry...'):
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]

            st.markdown("---")
            st.subheader("🎯 Classification Results")
            
            st.metric(label="Probability of Verification (โอกาสเป็นดาวเคราะห์จริง)", value=f"{probability:.2%}")
            if prediction[0] == 1:
                st.success("✅ **STATUS: CONFIRMED EXOPLANET**")
                st.progress(float(probability))
            else:
                st.error("❌ **STATUS: FALSE POSITIVE**")
                st.progress(float(1 - probability))

else:
    st.error("❌ Model artifact not found. Please train the model first.")