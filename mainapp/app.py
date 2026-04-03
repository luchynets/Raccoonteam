import streamlit as st
import tempfile
import sys
import os
import gc
import pandas as pd
import time

# --- НАЛАШТУВАННЯ ШЛЯХІВ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from parser.parser import TelemetryParser, TelemetryProcessor, UnitAutoDetector
    from metrics.metrics import calculate_metrics
    from visualization.plot import create_3d_plot
except ImportError as e:
    st.error(f"Помилка імпорту модулів: {e}")

# --- 1. КОНФІГУРАЦІЯ ТА СТИЛІ ---
st.set_page_config(page_title="UAV Analytics | Raccoonteam", page_icon="🚁", layout="wide")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 2.8rem; font-weight: 800; margin-bottom: 0.1rem; }
    .sub-title { text-align: center; font-size: 1.1rem; color: #A0AEC0; margin-bottom: 1.5rem; }
    .center-subheader { text-align: center; font-size: 1.5rem; font-weight: 600; margin: 1rem 0; color: #4CAF50; }
    
    [data-testid="stMetricValue"] { font-size: 1.7rem; justify-content: center; text-align: center; }
    [data-testid="stMetricLabel"] { justify-content: center; text-align: center; }
    
    .footer-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; padding: 30px; color: #718096; font-size: 14px;
        border-top: 1px solid #333; margin-top: 40px;
    }
    .center-info { text-align: center; margin-top: 4rem; color: #718096; }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>⚙️ Налаштування</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Завантажити лог (.bin)", type=['bin'])
    st.divider()
    if not uploaded_file:
        st.info("Чекаємо на файл для обробки...")

# --- 3. ГОЛОВНИЙ ЕКРАН ---
st.markdown("<h1 class='main-title'>🚁 Аналітика телеметрії БПЛА</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Система автоматичного розрахунку кінематики та візуалізації місій</p>", unsafe_allow_html=True)

st.divider()

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        with st.spinner('Обробка даних...'):
            # Обробка через вбудовані модулі команди
            tp = TelemetryParser(tmp_filepath)
            tp.parse() 
            gps_df, imu_df = tp.to_dataframe()

            if not gps_df.empty and not imu_df.empty:
                # Калібрування одиниць та мердж
                gps_df, _ = UnitAutoDetector.detect_gps(gps_df)
                imu_df, _ = UnitAutoDetector.detect_imu(imu_df)
                merged_df = TelemetryProcessor.merge(gps_df, imu_df)

                if merged_df is not None:
                    # Розрахунок основних метрик
                    metrics, enriched_data = calculate_metrics(merged_df)

                    # Візуалізація метрик
                    st.markdown("<h3 class='center-subheader'>📊 Показники місії</h3>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Дистанція", f"{metrics['total_distance']} м")
                    m2.metric("Час польоту", f"{metrics['flight_duration']} с")
                    m3.metric("Набір висоти", f"{metrics['max_altitude_gain']} м")

                    m4, m5, m6 = st.columns(3)
                    m4.metric("Макс. швидкість", f"{metrics['max_horizontal_speed']} м/с")
                    m5.metric("Верт. швидкість", f"{metrics['max_vertical_speed']} м/с")
                    m6.metric("Прискорення", f"{metrics['max_acceleration']} м/с²")
                    
                    st.divider()

                    # Побудова 3D траєкторії
                    st.markdown("<h3 class='center-subheader'>🗺️ 3D Траєкторія</h3>", unsafe_allow_html=True)
                    fig = create_3d_plot(enriched_data, metrics)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Не вдалося синхронізувати дані сенсорів.")
            else:
                st.error("У файлі недостатньо повідомлень GPS/IMU.")

    except Exception as e:
        st.error(f"Помилка: {e}")
    finally:
        gc.collect()
        try:
            if os.path.exists(tmp_filepath):
                time.sleep(0.2)
                os.remove(tmp_filepath)
        except:
            pass
else:
    st.markdown("<p class='center-info'>Чекаємо на завантаження лог-файлу для старту аналізу.</p>", unsafe_allow_html=True)

# --- 4. ФУТЕР ---
st.markdown("""
<div class="footer-container">
    <p>Made by <b>Raccoonteam</b></p>
</div>
""", unsafe_allow_html=True)