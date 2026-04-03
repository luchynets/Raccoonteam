import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from scipy.spatial.transform import Rotation as R

def calculate_metrics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Професійне ядро інерціальної навігації БПЛА (INS).
    
    Реалізовані методи:
        1. AHRS (Attitude and Heading Reference System): Використання кватерніонів для 3D-орієнтації.
        2. Gravity Compensation: Динамічне віднімання вектора g (9.81 м/с²) у світовій системі координат.
        3. Trapezoidal Integration: Чисельне інтегрування прискорення для отримання швидкості.
        4. Sensor Fusion: Комплементарний фільтр для об'єднання стабільності GPS та динаміки IMU.
    
    Повертає tuple: (metrics, data)
    """
    data = df.copy()

    # ---------------------------------------------------------
    # 1. ПІДГОТОВКА ТА ОЧИЩЕННЯ ДАНИХ
    # ---------------------------------------------------------
    # Заповнюємо пропуски сенсорів методами ffill/bfill для стабільності математики
    data[['lat', 'lon', 'alt']] = data[['lat', 'lon', 'alt']].ffill().bfill()
    
    # Групуємо дубльовані timestamp-и (характерно для логів ArduPilot)
    data = data.groupby('time').mean().reset_index()

    # Розрахунок дельта-часу (dt) в секундах
    data['dt'] = data['time'].diff() / 1e6
    data['dt'] = data['dt'].replace(0, np.nan).bfill().fillna(0)

    # ---------------------------------------------------------
    # 2. GPS АНАЛІТИКА (Векторизований Haversine)
    # ---------------------------------------------------------
    coords_prev = data[['lat', 'lon']].shift(1).bfill().values
    coords_now = data[['lat', 'lon']].values

    # Відстань між точками (метри)
    data['step_distance'] = haversine_vector(
        coords_prev, coords_now, Unit.METERS, comb=False
    )

    # Швидкість по GPS з обмеженням аномальних викидів (Outlier clipping)
    data['v_gps'] = (data['step_distance'] / data['dt']).replace([np.inf, -np.inf], 0).fillna(0).clip(upper=60)

    # ---------------------------------------------------------
    # 3. 3D-ОРІЄНТАЦІЯ ТА КВАТЕРНІОНИ (AHRS)
    # ---------------------------------------------------------
    # Initial Alignment: Визначаємо початковий нахил корпусу відносно вектора гравітації
    start_acc = np.array(data[['acc_x', 'acc_y', 'acc_z']].iloc[:10].mean().values, copy=True)
    acc_norm = np.linalg.norm(start_acc)

    # Створюємо об'єкт повороту (Rotation), вирівнюючи локальну Z-вісь з глобальною вертикаллю
    rotation = (
        R.align_vectors([[0, 0, 1]], [start_acc / acc_norm])[0]
        if acc_norm > 0.1
        else R.from_quat([0, 0, 0, 1])
    )

    acc_world = []      # Лінійне прискорення у світовій системі (ENU/NED)
    acc_mag_list = []   # Модуль прискорення

    # Ітеративне оновлення орієнтації через інтегрування кутової швидкості (Гіроскоп)
    for i in range(len(data)):
        if i > 0:
            # Кутова швидкість (rad/s)
            omega = np.array(data[['gyro_x', 'gyro_y', 'gyro_z']].iloc[i].values, copy=True)
            # Оновлення кватерніона повороту
            rotation = rotation * R.from_rotvec(omega * data['dt'].iloc[i])
            rotation = R.from_quat(rotation.as_quat() / np.linalg.norm(rotation.as_quat()))

        # Прискорення в системі БПЛА
        acc = np.array(data[['acc_x', 'acc_y', 'acc_z']].iloc[i].values, copy=True)
        # Трансформація вектора у світову систему та компенсація гравітації
        acc_w = rotation.apply(acc) - [0, 0, 9.81]

        acc_world.append(acc_w)
        acc_mag_list.append(np.linalg.norm(acc_w))

    acc_world = np.array(acc_world)
    data['acc_x_world'], data['acc_y_world'], data['acc_z_world'] = acc_world.T
    data['acc_mag_lin'] = acc_mag_list

    # ---------------------------------------------------------
    # 4. ТРАПЕЦІЄВИДНЕ ІНТЕГРУВАННЯ (IMU Velocity)
    # ---------------------------------------------------------
    # Розрахунок швидкості на основі акселерометра для фіксації миттєвих маневрів
    vx = np.zeros(len(data))
    vy = np.zeros(len(data))

    for i in range(1, len(data)):
        dt = data['dt'].iloc[i]
        # v = v_prev + 0.5 * (a_curr + a_prev) * dt
        vx[i] = vx[i-1] + (data['acc_x_world'].iloc[i] + data['acc_x_world'].iloc[i-1]) / 2 * dt
        vy[i] = vy[i-1] + (data['acc_y_world'].iloc[i] + data['acc_y_world'].iloc[i-1]) / 2 * dt

    data['v_imu'] = np.sqrt(vx**2 + vy**2)

   # ---------------------------------------------------------
    # 5. SENSOR FUSION (Адаптивний комплементарний фільтр)
    # ---------------------------------------------------------
    # Розраховуємо коефіцієнт довіри до швидкості GPS (від 0 до 1)
    # Якщо швидкість низька (дрон тупить/висить) — довіра до GPS падає.
    speed_conf = data['v_gps'].clip(0, 20) / 20 
    
    # Адаптивна альфа: базова 0.02 + надбавка залежно від швидкості
    # Це дозволяє динамічно змінювати вагу IMU в розрахунках
    data['alpha_dyn'] = 0.02 + 0.05 * speed_conf
    
    # Виконуємо змішування (Fusion)
    data['fused_speed'] = (1 - data['alpha_dyn']) * data['v_gps'] + data['alpha_dyn'] * data['v_imu']
    
    # Згладжування ковзним вікном для фільтрації високочастотного шуму
    data['fused_speed'] = data['fused_speed'].rolling(window=5, min_periods=1).mean()
    # ---------------------------------------------------------
    # 6. ВЕРТИКАЛЬНА АНАЛІТИКА ТА ВИСОТА
    # ---------------------------------------------------------
    # Фільтрація вертикальних стрибків GPS
    data['delta_alt'] = data['alt'].diff().fillna(0)
    data.loc[data['delta_alt'].abs() > 10, 'delta_alt'] = 0

    # Кумулятивний набір висоти (сума всіх підйомів)
    max_alt_gain = data['delta_alt'].clip(lower=0).sum()

    # Вертикальна швидкість на основі барометра/GPS
    v_ver = (data['delta_alt'] / data['dt']).replace([np.inf, -np.inf], 0).fillna(0)
    v_ver = v_ver.abs().clip(upper=30).rolling(window=10, min_periods=1).mean()

    # ---------------------------------------------------------
    # 7. ФОРМУВАННЯ ПІДСУМКОВОГО ЗВІТУ
    # ---------------------------------------------------------
    metrics = {
        "max_horizontal_speed": float(round(data['fused_speed'].max(), 2)),
        "max_vertical_speed": float(round(v_ver.max(), 2)),
        "max_acceleration": float(round(data['acc_mag_lin'].max(), 2)),
        "max_altitude_gain": float(round(max_alt_gain, 2)),
        "total_distance": float(round(data['step_distance'].sum(), 2)),
        "flight_duration": float(round((data['time'].max() - data['time'].min()) / 1e6, 2))
    }

    return metrics, data
