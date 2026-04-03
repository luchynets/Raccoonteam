import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
import pymap3d as pm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics.metrics import calculate_metrics

# Функція побудови графіка 
def create_3d_plot(df, metrics):
    # Очищуємо дані, якщо є нульові координати
    df = df[df['lat'] != 0].reset_index(drop=True)
    
    if df.empty or len(df) < 2:
        print("Критично: Недостатньо GPS даних для побудови шляху!")
        return go.Figure()

    lat0, lon0, alt0 = df['lat'].iloc[0], df['lon'].iloc[0], df['alt'].iloc[0]
    df['x'], df['y'], df['z'] = pm.geodetic2enu(
        df['lat'], df['lon'], df['alt'], 
        lat0, lon0, alt0
    )

    # Знаходимо найнижчу точку для правильної тіні
    z_floor = df['z'].min()
    
    fig = go.Figure()

    # Траєкторія
    fig.add_trace(go.Scatter3d(
        x=df['x'], 
        y=df['y'], 
        z=df['z'],
        mode='lines',
        line=dict(
            color=df['fused_speed'].values,
            colorscale='Jet',
            cmin=df['fused_speed'].min(),
            cmax=df['fused_speed'].max(),
            width=8,
            colorbar=dict(title="м/с", x=0.9)
        ),
        name='Маршрут БПЛА',
        customdata=np.stack((df['fused_speed'], df['fused_vz'], df['dt']), axis=-1),
        hovertemplate=(
            "<b>Висота:</b> %{z:.1f} м<br>" +
            "<b>Гор. швидкість:</b> %{customdata[0]:.2f} м/с<br>" +
            "<b>Верт. швидкість:</b> %{customdata[1]:.2f} м/с<br>" +
            "<extra></extra>"
        )
    ))

    # Тінь
    fig.add_trace(go.Scatter3d(
        x=df['x'], y=df['y'], z=[z_floor]*len(df),
        mode='lines',
        line=dict(color='rgba(150,150,150,0.5)', width=2, dash='dash'),
        name='Проекція',
        showlegend=False
    ))

    # Вертикальні лінії від тіні до траєкторії
    for i in range(0, len(df), 50):
        fig.add_trace(go.Scatter3d(
            x=[df['x'].iloc[i], df['x'].iloc[i]],
            y=[df['y'].iloc[i], df['y'].iloc[i]],
            z=[z_floor, df['z'].iloc[i]],
            mode='lines',
            line=dict(color='rgba(100,100,100,0.2)', width=1),
            showlegend=False, hoverinfo='skip'
        ))

    # Точка зльоту
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[0]], y=[df['y'].iloc[0]], z=[df['z'].iloc[0]],
        mode='markers', marker=dict(size=10, color='lime', symbol='circle'),
        name='СТАРТ'
    ))
    
    # Точка посадки
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], z=[df['z'].iloc[-1]],
        mode='markers', marker=dict(size=12, color='red', symbol='diamond'),
        name='ФІНІШ'
    ))

    # Метрики
    report = (
        f"<b>ЗВІТ ПОЛЬОТНОЇ МІСІЇ</b><br>"
        f"<br>Час у польоті: {metrics.get('flight_duration', 0):.1f} с"
        f"<br>Дистанція: {metrics.get('total_distance', 0):.1f} м"
        f"<br>Набір висоти: {metrics.get('max_altitude_gain', 0):.1f} м"
        f"<br>Макс. горз. швидкість: {metrics.get('max_horizontal_speed', 0):.2f} м/с"
        f"<br>Макс. верт. швидкість: {metrics.get('max_vertical_speed', 0):.2f} м/с"
        f"<br>Макс. прискорення: {metrics.get('max_acceleration', 0):.1f} м/с²"
    )

    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='East (м)',
            yaxis_title='North (м)',
            zaxis_title='Altitude (м)',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        annotations=[dict(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=report, showarrow=False, align="left",
            font=dict(size=13, color="white"),
            bgcolor="rgba(30, 30, 30, 0.85)",
            bordercolor="#444", borderwidth=1, borderpad=10
        )]
    )
    
    return fig

# Запуск
def main():
    # 1. Знаходимо файл (це залишається, бо нам треба звідкись брати сирі дані)
    csv_path = os.path.join("parser", "merged_output_01.csv")
    
    if not os.path.exists(csv_path):
        print(f"Помилка: Файл {csv_path} не знайдено!")
        return

    # 2. Читаємо СИРІ дані
    print(f"Зчитування даних з {csv_path}...")
    df_raw = pd.read_csv(csv_path)

    # 3. Віддаємо сирі дані в твоє ядро навігації!
    print("Обчислення метрик та Sensor Fusion...")
    metrics_result, processed_data = calculate_metrics(df_raw)

    # Збереження файлу (в основній гілці видалити)
    output_path = "processed_data_check1.csv"
    processed_data.to_csv(output_path, index=False)
    print(f"Оновлені дані успішно збережено у файл: {output_path}")
    # ----------------------------------
    
    # 4. Передаємо ПРОКАЧАНИЙ датафрейм (processed_data) у графік
    print("Побудова 3D графіка...")
    fig = create_3d_plot(processed_data, metrics_result)
    
    # Показуємо результат
    fig.show()

if __name__ == "__main__":
    main()