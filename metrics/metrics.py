import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from scipy.spatial.transform import Rotation as R


def calculate_metrics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Ядро інерціальної навігації БПЛА (INS).

    ---------------------------------------------------------
    МАТЕМАТИЧНЕ ОБҐРУНТУВАННЯ ТА МЕТОДОЛОГІЯ:
    ---------------------------------------------------------
    1. ОРІЄНТАЦІЯ (Quaternions vs Euler):
       Для відстеження положення ми використовуємо КВАТЕРНІОНИ. На відміну від кутів
       Ейлера (Roll/Pitch/Yaw), вони не мають проблеми "Gimbal Lock" (втрата ступеня
       свободи при нахилі 90°). Це критично для дронів, що здійснюють різкі маневри.
       Ми також впровадили примусову нормалізацію кватерніона на кожному кроці
       для компенсації чисельного дрейфу (numerical drift).

    2. КОМПЕНСАЦІЯ ГРАВІТАЦІЇ (Gravity Removal):
       Сирі дані акселерометра завжди містять вектор g (9.81 м/с²). Для отримання
       чистого лінійного прискорення ми трансформуємо вектор з локальної системи
       координат БПЛА (Body Frame) у світову (World Frame) і віднімаємо вектор [0,0,9.81].

    3. ПОХИБКИ ІНТЕГРУВАННЯ (Double Integration Error):
       Чисельне інтегрування прискорення для отримання швидкості (v = ∫a dt)
       накопичує похибку лінійно, а позиції (s = ∫v dt) — квадратично.
       Для мінімізації похибки ми використовуємо метод ТРАПЕЦІЙ та Sensor Fusion.

    4. SENSOR FUSION (Complementary Filter):
       Для подолання дрейфу IMU ми змішуємо його дані з GPS. Ми застосували
       АДАПТИВНУ АЛЬФУ: при високій швидкості (висока довіра до GPS) вага IMU
       змінюється, дозволяючи точніше фіксувати динамічні ривки.
    ---------------------------------------------------------
    """
    data = df.copy()

    # 1. ПІДГОТОВКА ДАНИХ
    data[['lat', 'lon', 'alt']] = data[['lat', 'lon', 'alt']].ffill().bfill()
    data = data.groupby('time').mean().reset_index()

    data['dt'] = data['time'].diff() / 1e6
    data['dt'] = data['dt'].fillna(data['dt'].median())
    # 2. GPS АНАЛІТИКА
    p_prev = data[['lat', 'lon']].shift(1).bfill().values
    p_curr = data[['lat', 'lon']].values

    # Використовуємо векторизацію для швидкої роботи на великих масивах даних
    data['step_dist'] = haversine_vector(p_prev, p_curr, Unit.METERS, comb=False)
    data['step_dist'] = data['step_dist'].clip(lower=0, upper=50)
    data['v_gps'] = (data['step_dist'] / data['dt']).replace([np.inf, -np.inf], 0).fillna(0)
    data['v_gps'] =  data['v_gps'].clip(upper=60)

    # 3. AHRS ТА ЛІНІЙНЕ ПРИСКОРЕННЯ
    init_acc = data[['acc_x', 'acc_y', 'acc_z']].iloc[:10].mean().values
    acc_norm = np.linalg.norm(init_acc)

    # Initial Alignment (початкове вирівнювання горизонту)
    # Створюємо об'єкт повороту (Rotation), вирівнюючи локальну Z-вісь з глобальною вертикаллю
    rotation = R.align_vectors([[0, 0, 1]], [init_acc / acc_norm])[0] if acc_norm > 0.1 else R.from_quat([0, 0, 0, 1])

    acc_w_list = np.zeros((len(data), 3))
    g_const = np.array([0, 0, 9.81])

    for i in range(len(data)):
        if i > 0:
            omega = np.array(data[['gyro_x', 'gyro_y', 'gyro_z']].iloc[i].values, copy=True)
            # Оновлення орієнтації (Integration of angular velocity)
            rotation = rotation * R.from_rotvec(omega * data['dt'].iloc[i])
            # Нормалізація для стабільності
            q = rotation.as_quat()
            rotation = R.from_quat(q / np.linalg.norm(q))

        # Перетворення Body -> World Frame
        raw_a = np.array(data[['acc_x', 'acc_y', 'acc_z']].iloc[i].values, copy=True)
        acc_w_list[i] = rotation.apply(raw_a) - g_const

    data[['ax_w', 'ay_w', 'az_w']] = acc_w_list
    data['acc_mag'] = np.linalg.norm(acc_w_list, axis=1)

    # 4. Векторизоване інтегрування (Trapezoidal Rule)
    # Знаходимо середнє прискорення між двома сусідніми точками (a_avg = (a_curr + a_prev) / 2)
    acc_x_mid = (data['ax_w'] + data['ax_w'].shift(1)) * 0.5
    acc_y_mid = (data['ay_w'] + data['ay_w'].shift(1)) * 0.5
    acc_z_mid = (data['az_w'] + data['az_w'].shift(1)) * 0.5

    # Множимо кожне середнє прискорення на його dt (отримуємо приріст швидкості dv)
    # fillna(0) потрібен для першого рядка, де немає попереднього значення
    dv_x = (acc_x_mid * data['dt']).fillna(0)
    dv_y = (acc_y_mid * data['dt']).fillna(0)
    dv_z = (acc_z_mid * data['dt']).fillna(0)

    vx = dv_x.cumsum()
    vy = dv_y.cumsum()

    data['v_imu'] = np.sqrt(vx ** 2 + vy ** 2)

    # 5. ADAPTIVE SENSOR FUSION
    # Адаптивна альфа: базова 0.02 + надбавка залежно від швидкості
    # Це дозволяє динамічно змінювати вагу IMU в розрахунках
    speed_conf = data['v_gps'].clip(0, 20) / 20
    alpha = 0.02 + 0.05 * speed_conf
    data['fused_speed'] = (1 - alpha) * data['v_gps'] + alpha * data['v_imu']
    data['fused_speed'] = data['fused_speed'].rolling(5, min_periods=1).mean()

    # --- 6. ВЕРТИКАЛЬНА АНАЛІТИКА ТА ADAPTIVE SENSOR FUSION ---
    v_imu_z = dv_z.cumsum()

    # GPS-складова (фільтрована дельта висоти)
    alt_smooth = data['alt'].rolling(window=10, min_periods=1, center=True).median()
    data['delta_alt'] = alt_smooth.diff().fillna(0)

    # Швидкість по GPS (dz/dt) з обмеженням шумів
    v_gps_z = (data['delta_alt'] / (data['dt'] + 1e-6)).fillna(0).clip(-20, 20)

    # ADAPTIVE vertical fusion:
    # GPS altitude is noisy, but IMU drifts vertically over time.
    # We trust IMU MORE during high vertical acceleration (climbs/dives),
    # and trust GPS MORE during stable flight (low vertical acceleration).
    # 
    # Strategy: Alpha increases with vertical acceleration magnitude
    vertical_accel = data['az_w'].abs()
    # High trust in IMU (alpha=0.15) during aggressive climbs/dives
    # Low trust in IMU (alpha=0.03) during stable flight
    alpha_z = 0.03 + 0.12 * (vertical_accel.clip(0, 10) / 10)
    
    data['fused_vz'] = (1 - alpha_z) * v_gps_z + alpha_z * v_imu_z
    data['fused_vz'] = data['fused_vz'].rolling(window=5, min_periods=1).mean()

    # 7. Формування метрик
    metrics = {
        "max_horizontal_speed": float(round(data['fused_speed'].max(), 2)),
        "avg_horizontal_speed": float(round(data['fused_speed'].mean(), 2)),
        "max_vertical_speed": float(round(data['fused_vz'].abs().max(), 2)),
        "avg_vertical_speed": float(round(data['fused_vz'].abs().mean(), 2)),
        "max_acceleration": float(round(data['acc_mag'].max(), 2)),
        "avg_acceleration": float(round(data['acc_mag'].mean(), 2)),
        "max_altitude_gain": float(round(data['delta_alt'].clip(lower=0).sum(), 2)),
        "total_distance": float(round(data['step_dist'].sum(), 2)),
        "flight_duration": float(round((data['time'].max() - data['time'].min()) / 1e6, 2))
    }

    return metrics, data


