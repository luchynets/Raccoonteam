import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pymap3d as pm

# АНАЛІТИЧНЕ ЯДРО 
def calculate_haversine_distance(lat, lon):
    """
    Обчислення відстані Haversine. 
    Рахує реальний шлях по кривій поверхні Землі.
    """
    R = 6371000 
    

    phi1 = np.radians(lat.shift(1))
    phi2 = np.radians(lat)
    dphi = np.radians(lat - lat.shift(1))
    dlambda = np.radians(lon - lon.shift(1))
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def trapezoidal_velocity(acc_array, dt_array):
    """
    Інтегрування прискорення для отримання швидкості (метод трапецій).
    """
    avg_acc = (acc_array + acc_array.shift(1)) / 2.0
    delta_v = avg_acc * dt_array
    return delta_v.cumsum().fillna(0)

def calculate_flight_metrics(df):
    m = {}

    # ЧАС 
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    dt = df['time'].diff().fillna(0)
    
    median_dt = dt.median()
    if median_dt > 10000:
        dt_sec = dt / 1_000_000
    elif median_dt > 10:
        dt_sec = dt / 1000
    else:
        dt_sec = dt

    df['dt_sec'] = dt_sec
    m['duration'] = df['dt_sec'].sum()

    # ДИСТАНЦІЯ 
    step_h = calculate_haversine_distance(df['lat'], df['lon']).fillna(0)
    step_h = step_h.mask(step_h > 100, 0)
    
    step_z = df['z'].diff().fillna(0)
    step_3d = np.sqrt(step_h**2 + step_z**2)
    m['total_dist'] = step_3d.sum()
    m['max_alt'] = df['z'].max()

    # ШВИДКІСТЬ 
    dt_s = df['dt_sec'].replace(0, np.nan) 
    
    vx = df['x'].diff() / dt_s
    vy = df['y'].diff() / dt_s
    vz = df['z'].diff() / dt_s

    df['v_hor'] = np.sqrt(vx**2 + vy**2).fillna(0)
    df['v_ver'] = vz.fillna(0)
    df.loc[df['v_hor'] > 60, 'v_hor'] = 0 

    m['max_v_hor'] = df['v_hor'].max()
    m['max_v_ver'] = df['v_ver'].abs().max()
    
    if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
        v_int_x = trapezoidal_velocity(df['acc_x'], df['dt_sec'])
        v_int_y = trapezoidal_velocity(df['acc_y'], df['dt_sec'])
        v_int_z = trapezoidal_velocity(df['acc_z'] - 9.81, df['dt_sec']) # Віднімаємо гравітацію!
        
        df['v_integrated'] = np.sqrt(v_int_x**2 + v_int_y**2 + v_int_z**2)
        m['max_v_integrated'] = df['v_integrated'].max()

    # ПРИСКОРЕННЯ ТА СТАТУС 
    if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
        g_vector = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

        g_filtered = g_vector.rolling(window=15, center=True).median().fillna(g_vector.median())
        
        acc_clean = (g_filtered - 9.81).abs()
        

        max_val = acc_clean.max()
        if max_val > 50:
             acc_clean = acc_clean / 5 
        
        m['max_acc'] = acc_clean.max()
    else:
        m['max_acc'] = 0.0

    if m['max_acc'] < 4:
        m['status'] = "СТАБІЛЬНО"
    elif m['max_acc'] < 15:
        m['status'] = "МАНЕВРОВИЙ ПОЛІТ"
    else:
        m['status'] = "ПЕРЕВАНТАЖЕННЯ / СУРОВІ УМОВИ"

    return m


# ВІЗУАЛІЗАЦІЯ 
def create_3d_plot(df_input):
    df = df_input.dropna(subset=['lat', 'lon', 'alt']).copy()
    df = df.drop_duplicates(subset=['time']).copy()
    df = df[df['lat'] != 0].reset_index(drop=True)
    print(f"Унікальних значень LAT: {df['lat'].nunique()}, LON: {df['lon'].nunique()}")
    if df.empty or len(df) < 2:
        print("Критично: Недостатньо GPS даних для побудови шляху!")
        return go.Figure()

    lat0, lon0, alt0 = df['lat'].iloc[0], df['lon'].iloc[0], df['alt'].iloc[0]
    df['x'], df['y'], df['z'] = pm.geodetic2enu(
        df['lat'], df['lon'], df['alt'], 
        lat0, lon0, alt0
    )

    metrics = calculate_flight_metrics(df)
    
    fig = go.Figure()

    # Траєкторія
    fig.add_trace(go.Scatter3d(
        x=df['x'], 
        y=df['y'], 
        z=df['z'],
        mode='lines',
        line=dict(
            color=df['v_hor'], 
            colorscale='Jet',
            width=8,
            colorbar=dict(title="м/с", x=0.9)
        ),
        name='Маршрут БПЛА',
        customdata=np.stack((df['v_hor'], df['v_ver'], df['dt_sec']), axis=-1),
        hovertemplate=(
            "<b>Висота:</b> %{z:.1f} м<br>" +
            "<b>Горизонтальна швидкість:</b> %{customdata[0]:.2f} м/с<br>" +
            "<b>Вертикальна швидкість:</b> %{customdata[1]:.2f} м/с<br>" +
            "<extra></extra>"
        )
    ))

    # Тінь
    fig.add_trace(go.Scatter3d(
        x=df['x'], y=df['y'], z=[0]*len(df),
        mode='lines',
        line=dict(color='rgba(150,150,150,0.5)', width=2, dash='dash'),
        name='Проекція',
        showlegend=False
    ))

    for i in range(0, len(df), 50):
        fig.add_trace(go.Scatter3d(
            x=[df['x'].iloc[i], df['x'].iloc[i]],
            y=[df['y'].iloc[i], df['y'].iloc[i]],
            z=[0, df['z'].iloc[i]],
            mode='lines',
            line=dict(color='rgba(100,100,100,0.2)', width=1),
            showlegend=False, hoverinfo='skip'
        ))

    # Точки зльоту та посадки
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[0]], y=[df['y'].iloc[0]], z=[df['z'].iloc[0]],
        mode='markers', marker=dict(size=10, color='lime', symbol='circle'),
        name='СТАРТ'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], z=[df['z'].iloc[-1]],
        mode='markers', marker=dict(size=12, color='red', symbol='diamond'),
        name='ФІНІШ'
    ))

    # ОФОРМЛЕННЯ ТА МЕТРИКИ 
    report = (
        f"<b>ЗВІТ ПОЛЬОТНОЇ МІСІЇ</b><br>"
        f"<br>Час у польоті: {metrics['duration']:.1f} с"
        f"<br>Дистанція: {metrics['total_dist']:.1f} м"
        f"<br>Макс. висота: {metrics['max_alt']:.1f} м"
        f"<br>Макс. горз. швидкість: {metrics['max_v_hor']:.2f} м/с"
        f"<br>Макс. верт. швидкість: {metrics['max_v_ver']:.2f} м/с"
        f"<br>Макс. прискорення: {metrics['max_acc']:.1f} м/с²"
        f"<br>Статус: <b>{metrics['status']}</b>"
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


if __name__ == "__main__":
    path = 'gps_data2.csv'
    
    if os.path.exists(path):
        print(f"Завантаження даних з {path}...")
        
        try:
            df_raw = pd.read_csv(path)
            
            required = ['time', 'lat', 'lon', 'alt']
            if all(col in df_raw.columns for col in required):
                fig = create_3d_plot(df_raw)
                fig.show()
            else:
                print(f"Помилка: У файлі відсутні колонки {required}")
        except Exception as e:
            print(f"Помилка при читанні: {e}")
    else:
        print(f"Файл {path} не знайдено!")
