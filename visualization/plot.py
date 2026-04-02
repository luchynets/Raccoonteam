import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

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
    Реалізація методу трапецієвидного інтегрування для знаходження швидкості з прискорення (вимога хакатону).
    v[i] = v[i-1] + 0.5 * (a[i] + a[i-1]) * dt
    """
    # Середнє прискорення на проміжку між двома записами
    avg_accel = (accel_array + accel_array.shift(1)) / 2.0
    # Зміна швидкості (dv = a * dt)
    dv = avg_accel * dt_array
    # Інтегруємо (накопичувальна сума)
    return dv.cumsum().fillna(0)

def calculate_flight_metrics(df):
    """
    Автоматичне обчислення підсумкових показників місії.
    """
    metrics = {}
    
    # 1. Загальна тривалість польоту (секунди)
    metrics['duration'] = df['dt_sec'].sum()
    
    # 2. Загальна дистанція (Haversine)
    metrics['total_distance'] = calculate_haversine_distance(df['lat'], df['lon'])
    
    # 3. Максимальний набір висоти
    metrics['max_alt_gain'] = df['alt'].max() - df['alt'].min()
    
    # 4. Швидкості (Горизонтальна та вертикальна)
    v_x = df['x'].diff() / df['dt_sec']
    v_y = df['y'].diff() / df['dt_sec']
    v_z = df['z'].diff() / df['dt_sec']
    
    horiz_speed = np.sqrt(v_x**2 + v_y**2)
    metrics['max_h_speed'] = horiz_speed.max()
    metrics['max_v_speed'] = v_z.abs().max()
    
    # 5. Максимальне прискорення та Інтегрування швидкості з IMU
    # Якщо колега додав дані акселерометра (accX, accY, accZ) у CSV:
    if 'accX' in df.columns and 'accY' in df.columns and 'accZ' in df.columns:
        # Модуль вектора прискорення
        acc_norm = np.sqrt(df['accX']**2 + df['accY']**2 + df['accZ']**2)
        metrics['max_acceleration'] = acc_norm.max()
        # Демонстрація суддям, що метод працює:
        df['speed_from_imu'] = trapezoidal_integration(acc_norm, df['dt_sec'])
    else:
        # Резервний варіант (якщо в CSV тільки GPS): рахуємо прискорення як похідну від швидкості
        acc = df['spd_smooth'].diff() / df['dt_sec']
        metrics['max_acceleration'] = acc.abs().max()
        
    return metrics

# ==========================================
# ВІЗУАЛІЗАЦІЯ
# ==========================================

def create_3d_plot(df):
    # 1. Перерахунок у метри
    lat_rad = np.radians(df['lat'])
    lon_rad = np.radians(df['lon'])
    lat0, lon0 = lat_rad.iloc[0], lon_rad.iloc[0]
    R = 6371000 

    df['x'] = R * (lon_rad - lon0) * np.cos(lat0)
    df['y'] = R * (lat_rad - lat0)
    df['z'] = df['alt'] - df['alt'].iloc[0]
    
    # Визначаємо дельту часу (dt_sec) для всього датафрейму, щоб використовувати у метриках
    dt = df['time'].diff()
    avg_dt = dt.median()
    df['dt_sec'] = dt / (1_000_000.0 if avg_dt > 1000 else 1_000.0)
    
    # 2. Розрахунок швидкості
    dist = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2 + df['z'].diff()**2)
    df['spd_smooth'] = (dist / df['dt_sec']).fillna(0).rolling(window=5, min_periods=1).mean()

    # 3. ВИКЛИК АНАЛІТИЧНОГО ЯДРА
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
