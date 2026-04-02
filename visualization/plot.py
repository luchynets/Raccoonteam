import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

def create_3d_plot(df):
    # 1. Перерахунок у метри (з підвищеною точністю)
    # Переводимо все в радіани одразу
    lat_rad = np.radians(df['lat'])
    lon_rad = np.radians(df['lon'])
    
    lat0 = lat_rad.iloc[0]
    lon0 = lon_rad.iloc[0]
    
    R = 6371000 # Радіус Землі в метрах

    # Формула проекції (локальна декартова система)
    df['x'] = R * (lon_rad - lon0) * np.cos(lat0)
    df['y'] = R * (lat_rad - lat0)
    df['z'] = df['alt'] - df['alt'].iloc[0]
    
    # ПЕРЕВІРКА: Виведемо в консоль максимальні значення, щоб переконатися, що вони не 0
    print(f"Max X: {df['x'].max()} m, Max Y: {df['y'].max()} m")

    # 2. Розрахунок швидкості з виправленням таймінгу
    # Обчислюємо відстань між сусідніми точками
    dist = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2 + df['z'].diff()**2)
    
    # Різниця в часі
    dt = df['time'].diff()
    
    # Авто-визначення: якщо dt занадто велике, то це мікросекунди, якщо мале - мілісекунди
    # Ми припускаємо, що між записами проходить від 0.01 до 0.5 секунди
    avg_dt = dt.median()
    if avg_dt > 1000:
        dt_seconds = dt / 1_000_000.0  # Мікросекунди -> Секунди
    else:
        dt_seconds = dt / 1_000.0      # Мілісекунди -> Секунди

    # Рахуємо швидкість м/с
    df['spd'] = (dist / dt_seconds).fillna(0)
    
    # Згладжування (ковзне середнє по 5 точках), щоб графік був плавний
    df['spd_smooth'] = df['spd'].rolling(window=5, min_periods=1).mean()

    fig = go.Figure()

    # 3. Траєкторія
    fig.add_trace(go.Scatter3d(
        x=df['x'], y=df['y'], z=df['z'],
        mode='lines',
        line=dict(
            color=df['spd_smooth'], 
            colorscale='Turbo',    
            width=6,               
            colorbar=dict(title="Швидкість (м/с)", thickness=15)
        ),
        # Додаємо Схід (x), Північ (y) та Висоту (z)
        hovertemplate=(
            "<b>Координати:</b><br>" +
            "Схід: %{x:.1f} м<br>" +
            "Північ: %{y:.1f} м<br>" +
            "Висота: %{z:.1f} м<br>" +
            "<b>Швидкість:</b> %{customdata:.2f} м/с" +
            "<extra></extra>"
        ),
        customdata=df['spd_smooth']
    ))

    # 4. Тінь
    fig.add_trace(go.Scatter3d(
        x=df['x'], y=df['y'], z=[0] * len(df),
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.2)', width=3),
        hoverinfo='skip'
    ))

    # 5. Дизайн (Змінено aspectmode на 'cube', щоб бачити об'єм, якщо шлях малий)
    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='Схід (м)',
            yaxis_title='Північ (м)',
            zaxis_title='Висота (м)',
            aspectmode='cube' 
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folder, 'gps_data.csv')
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not df.empty:
            fig = create_3d_plot(df)
            fig.show()
