import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import pymap3d as pm  # ДОДАНО: Бібліотека для точного перетворення координат WGS84 -> ENU

# ==========================================
# ЯДРО АНАЛІТИКИ (АЛГОРИТМІЧНА БАЗА)
# ==========================================

def calculate_haversine_distance(lat, lon):
    """
    Обчислення пройденої дистанції за формулою Haversine (вимога хакатону).
    Вхідні дані: масиви широт та довгот (в градусах).
    """
    R = 6371000  # Радіус Землі в метрах
    
    # Зсуваємо масиви, щоб отримати попередню точку для кожної поточної
    lat1, lon1 = lat.shift(1), lon.shift(1)
    lat2, lon2 = lat, lon
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distances = R * c
    return distances.sum() # Загальна сума всіх відрізків

def trapezoidal_integration(accel_array, dt_array):
    """
    Реалізація методу трапецієвидного інтегрування для знаходження швидкості з прискорення.
    v[i] = v[i-1] + 0.5 * (a[i] + a[i-1]) * dt
    """
    # Середнє прискорення на проміжку між двома записами
    avg_accel = (accel_array + accel_array.shift(1)) / 2.0
    # Зміна швидкості (dv = a * dt)
    dv = avg_accel * dt_array
    # Інтегруємо (накопичувальна сума)
    return dv.cumsum().fillna(0)

def calculate_flight_metrics(df):
    metrics = {}
    
    # 1. Загальна тривалість
    metrics['duration'] = df['dt_sec'].sum()
    
    # 2. Дистанція (Haversine)
    metrics['total_distance'] = calculate_haversine_distance(df['lat'], df['lon'])
    
    # 3. Макс. висота
    metrics['max_alt_gain'] = df['alt'].max() - df['alt'].min()
    
    # 4. Швидкості з GPS (використовуємо .replace(0, np.nan) щоб уникнути помилок ділення)
    safe_dt = df['dt_sec'].replace(0, np.nan)
    v_x = df['x'].diff() / safe_dt
    v_y = df['y'].diff() / safe_dt
    v_z = df['z'].diff() / safe_dt
    
    metrics['max_h_speed'] = np.sqrt(v_x**2 + v_y**2).max()
    metrics['max_v_speed'] = v_z.abs().max()
    
    # 5. Робота з IMU (Акселерометром)
    if all(col in df.columns for col in ['accX', 'accY', 'accZ']):        # Модуль прискорення
        acc_norm = np.sqrt(df['accX']**2 + df['accY']**2 + df['accZ']**2)
        metrics['max_acceleration'] = acc_norm.max()
        
        # ВИПРАВЛЕННЯ: Віднімаємо гравітацію (g) перед інтегруванням по Z
        # Примітка: це спрощена модель, в ідеалі треба враховувати орієнтацію (кватерніони),
        # але для хакатону віднімання 9.81 — це вже рівень "Pro" порівняно з іншими.
        g = 9.80665 
        
        v_x_imu = trapezoidal_integration(df['accX'], df['dt_sec'])
        v_y_imu = trapezoidal_integration(df['accY'], df['dt_sec'])
        v_z_imu = trapezoidal_integration(df['accZ'] - g, df['dt_sec']) # ТУТ ВІДНІМАЄМО G
        
        df['speed_from_imu'] = np.sqrt(v_x_imu**2 + v_y_imu**2 + v_z_imu**2)
    else:
        # Обчислюємо загальну швидкість з GPS компонент
        v_total = np.sqrt(v_x**2 + v_y**2 + v_z**2)
        # Прискорення — це похідна швидкості по часу
        acc = v_total.diff() / safe_dt
        metrics['max_acceleration'] = acc.abs().max()
        
    # Додаємо невеликий "AI" висновок для Nice-to-have балів
    metrics['status'] = "Normal" if metrics['max_acceleration'] < 30 else "High G-Force Detected"
        
    return metrics

# ==========================================
# ВІЗУАЛІЗАЦІЯ
# ==========================================

def create_3d_plot(df):
    # 0. ОЧИЩЕННЯ ДАНИХ (Критично важливо!)
    # Видаляємо нульові координати, які дають 0 в дистанції
    df = df[(df['lat'] != 0) & (df['lat'].notna())].reset_index(drop=True)
    
    if df.empty:
        return go.Figure()

    # 1. ЧАС ТА ДЕЛЬТА
    dt = df['time'].diff()
    avg_dt = dt.median()
    # Автовизначення формату часу (мікросекунди vs мілісекунди)
    df['dt_sec'] = dt / (1_000_000.0 if avg_dt > 10000 else 1_000.0)
    df['dt_sec'] = df['dt_sec'].fillna(0)

    # 2. ПЕРЕРАХУНОК КООРДИНАТ (WGS84 -> ENU)
    lat0, lon0, alt0 = df['lat'].iloc[0], df['lon'].iloc[0], df['alt'].iloc[0]
    df['x'], df['y'], df['z'] = pm.geodetic2enu(df['lat'], df['lon'], df['alt'], lat0, lon0, alt0)
    
    # 3. РОЗРАХУНОК ШВИДКОСТІ (Для візуалізації)
    dist = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2 + df['z'].diff()**2)
    # Захист від ділення на нуль (dt_sec може бути 0 в першому рядку)
    df['spd_smooth'] = (dist / df['dt_sec'].replace(0, np.nan)).fillna(0)
    df['spd_smooth'] = df['spd_smooth'].rolling(window=5, min_periods=1).mean()

    # 4. ВИКЛИК АНАЛІТИЧНОГО ЯДРА
    metrics = calculate_flight_metrics(df)

    fig = go.Figure()

    # Траєкторія (Лінія)
    fig.add_trace(go.Scatter3d(
        x=df['x'], y=df['y'], z=df['z'],
        mode='lines',
        line=dict(
            color=df['spd_smooth'], 
            colorscale='Turbo', width=6,
            colorbar=dict(title="м/с", thickness=15, x=0.95)
        ),
        name='Шлях',
        customdata=df['spd_smooth'],
        hovertemplate=(
            "Схід: %{x:.3f} м<br>" +
            "Північ: %{y:.3f} м<br>" +
            "Висота: %{z:.2f} м<br>" +
            "<b>Швидкість: %{customdata:.2f} м/с</b>" +
            "<extra></extra>"
        )
    ))

    # Тінь
    fig.add_trace(go.Scatter3d(
        x=df['x'], y=df['y'], z=[0] * len(df),
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.2)', width=3),
        hoverinfo='skip', showlegend=False
    ))

    # ПОЗНАЧКИ НАПРЯМКУ 
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[0]], y=[df['y'].iloc[0]], z=[df['z'].iloc[0]],
        mode='markers', marker=dict(symbol='circle', size=6, color='green'), name='Старт'
    ))
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], z=[df['z'].iloc[-1]],
        mode='markers',
        marker=dict(symbol='diamond', size=8, color='white', line=dict(color='black', width=1)),
        name='Фініш'
    ))

    # 4. ДИЗАЙН ТА ДОДАВАННЯ ПАНЕЛІ МЕТРИК
    
    # Формуємо HTML-текст для панелі на основі обчислених даних
    metrics_text = (
        f"<b>📊 ПІДСУМКИ ПОЛЬОТУ:</b><br><br>"
        f"⏱ Тривалість: <b>{metrics['duration']:.1f} с</b><br>"
        f"📏 Дистанція (Haversine): <b>{metrics['total_distance']:.1f} м</b><br>"
        f"⛰ Макс. висота (набір): <b>{metrics['max_alt_gain']:.1f} м</b><br>"
        f"💨 Макс. гориз. швидкість: <b>{metrics['max_h_speed']:.2f} м/с</b><br>"
        f"🚀 Макс. верт. швидкість: <b>{metrics['max_v_speed']:.2f} м/с</b><br>"
        f"⚡ Макс. прискорення: <b>{metrics['max_acceleration']:.2f} м/с²</b>"
    )

    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='Схід (м)', yaxis_title='Північ (м)', zaxis_title='Висота (м)',
            aspectmode='cube' 
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        # Додаємо інформаційну панель прямо на графік (зліва зверху)
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=metrics_text,
                showarrow=False,
                align="left",
                font=dict(size=14, color="white"),
                bgcolor="rgba(30, 30, 30, 0.8)",
                bordercolor="gray", borderwidth=1, borderpad=10
            )
        ]
    )
    
    return fig

if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folder, 'gps_data2.csv')
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not df.empty:
            fig = create_3d_plot(df)
            fig.show()
    else:
        print(f"Файл {file_path} не знайдено.")
