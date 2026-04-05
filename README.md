# 🛸 UAV Telemetry Analytics: Navigation Core

A professional-grade UAV telemetry analysis system that combines GPS and IMU data using Inertial Navigation System (INS) principles and adaptive sensor fusion.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web Interface (Streamlit)](#web-interface-streamlit)
  - [Command Line Interface](#command-line-interface)
- [Analytics Core: INS Methodology](#analytics-core-ins-methodology)
  - [1. 3D Orientation & Quaternions (AHRS)](#1-3d-orientation--quaternions-ahrs)
  - [2. Gravity Compensation](#2-gravity-compensation)
  - [3. Trapezoidal Integration](#3-trapezoidal-integration)
  - [4. Adaptive Sensor Fusion](#4-adaptive-sensor-fusion)
- [Key Flight Metrics](#key-flight-metrics)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [License & Credits](#license--credits)

---

## 🎯 Overview

This project implements an intelligent UAV telemetry processing pipeline that transforms raw binary flight logs (MAVLink format) into actionable flight analytics. Unlike simple GPS trackers, our system leverages **Inertial Navigation System (INS)** principles to achieve superior accuracy by fusing GPS positioning with high-frequency IMU data.

**Key Differentiator**: Adaptive sensor fusion that dynamically adjusts sensor trust based on flight dynamics, providing accurate speed and altitude calculations even during aggressive maneuvers.

---

## ✨ Features

- 📊 **Comprehensive Flight Metrics**: Distance, duration, speed (horizontal/vertical), acceleration, altitude gain
- 🗺️ **3D Trajectory Visualization**: Interactive Plotly plots with speed-based color coding
- 🧮 **INS-Core Calculations**: Quaternion-based orientation, gravity compensation, trapezoidal integration
- 🔗 **Adaptive Sensor Fusion**: Dynamic GPS+IMU blending based on flight conditions
- 📈 **Automatic Unit Detection**: Handles various MAVLink data formats (deg×1e7, mm, milli-g, etc.)
- 🖥️ **Web Dashboard**: User-friendly Streamlit interface with drag-and-drop file upload
- 💻 **CLI Support**: Batch processing capability for automated workflows

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WEB INTERFACE (Streamlit)                │
│                       mainapp/app.py                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    PARSER MODULE                              │
│                  parser/parser.py                             │
│                                                               │
│  • TelemetryParser: Raw .bin → GPS + IMU DataFrames          │
│  • UnitAutoDetector: Auto-convert units (mm→m, deg*1e7→deg)  │
│  • TelemetryProcessor: Merge & synchronize sensors            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   METRICS MODULE                              │
│                 metrics/metrics.py                            │
│                                                               │
│  • AHRS: Quaternion-based 3D orientation                     │
│  • Gravity Compensation: Remove g from accelerometer          │
│  • Trapezoidal Integration: Velocity from acceleration        │
│  • Adaptive Sensor Fusion: GPS + IMU blending                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                VISUALIZATION MODULE                           │
│               visualization/plot.py                           │
│                                                               │
│  • 3D trajectory plot with Plotly                             │
│  • Geodetic → ENU coordinate conversion (pymap3d)            │
│  • Speed color coding & mission report overlay                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- **Python 3.10+**
- **pip** package manager

### Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/Raccoonteam
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - `pandas`, `numpy` - Data processing
   - `scipy` - Mathematical transformations & quaternions
   - `pymavlink`, `pymavlog` - MAVLink binary log parsing
   - `streamlit` - Web interface
   - `plotly` - Interactive 3D visualizations
   - `haversine` - Geodesic distance calculations
   - `pymap3d` - Coordinate system conversions

---

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

```bash
streamlit run mainapp/app.py
```

This opens a browser window at `http://localhost:8501` with:
- File uploader for `.bin` telemetry logs
- Real-time metrics dashboard
- Interactive 3D trajectory viewer

### Option 2: Command Line Processing

```bash
# Parse and merge GPS/IMU data
python parser/parser.py data/flight_log.BIN --output merged_output.csv

# Generate 3D visualization (standalone)
python visualization/plot.py
```

---

## 📖 Usage

### Web Interface (Streamlit)

1. **Launch the app**:
   ```bash
   streamlit run mainapp/app.py
   ```

2. **Upload a telemetry file**:
   - Click the sidebar uploader
   - Select a `.bin` MAVLink log file

3. **View results**:
   - 📊 **Mission Metrics Panel**: Distance, flight time, altitude, speed, acceleration
   - 🗺️ **3D Trajectory Plot**: Interactive visualization with hover tooltips

### Command Line Interface

#### Parser Module

```bash
# Basic usage
python parser/parser.py path/to/flight.BIN --output output.csv

# The parser will:
# 1. Extract GPS and IMU messages from the binary log
# 2. Auto-detect and normalize units
# 3. Compute sensor frequencies
# 4. Merge data into a synchronized CSV
```

#### Visualization Module

```bash
# Ensure parser/merged_output_01.csv exists first
python visualization/plot.py

# This will:
# 1. Read merged CSV data
# 2. Calculate metrics via metrics.calculate_metrics()
# 3. Generate interactive 3D plot
# 4. Save processed data to processed_data_check1.csv
```

---

## 🧮 Analytics Core: INS Methodology

### 1. 3D Orientation & Quaternions (AHRS)

**Problem**: Euler angles (roll/pitch/yaw) suffer from **Gimbal Lock** at 90° pitch, causing catastrophic orientation failures during aggressive maneuvers.

**Solution**: We use **quaternions** (`scipy.spatial.transform.Rotation`) for drift-free 3D orientation tracking.

```python
# Initial alignment using gravity vector
init_acc = data[['acc_x', 'acc_y', 'acc_z']].iloc[:10].mean().values
rotation = R.align_vectors([[0, 0, 1]], [init_acc / acc_norm])[0]

# Quaternion normalization prevents numerical drift
q = rotation.as_quat()
rotation = R.from_quat(q / np.linalg.norm(q))
```

### 2. Gravity Compensation

**Problem**: Accelerometers measure both **linear acceleration** AND **gravity** (9.81 m/s²). Raw data includes Earth's gravity vector.

**Solution**: Transform accelerometer data from body frame to world frame, then subtract gravity:

```python
# Body Frame → World Frame transformation
raw_a = np.array([acc_x, acc_y, acc_z])
acc_world = rotation.apply(raw_a) - np.array([0, 0, 9.81])
```

This isolates **pure linear acceleration** (motor thrust) from gravitational effects.

### 3. Trapezoidal Integration

**Problem**: Converting acceleration → velocity via integration accumulates errors. Simple rectangular integration (`v = a * dt`) is inaccurate for dynamic maneuvers.

**Solution**: We use the **trapezoidal rule** for higher accuracy:

```
v_i = v_{i-1} + (a_i + a_{i-1}) / 2 * Δt
```

```python
acc_mid = (acc_curr + acc_prev) * 0.5
dv = acc_mid * dt
velocity = dv.cumsum()
```

### 4. Adaptive Sensor Fusion

**Problem**: 
- **GPS**: Stable long-term but noisy and low-frequency (~5-10 Hz)
- **IMU**: High-frequency (~100-400 Hz) but drifts over time

**Solution**: Complementary filter with **adaptive alpha** that adjusts based on flight dynamics.

#### Horizontal Fusion

```python
# Trust IMU more during high-speed maneuvers
speed_conf = v_gps.clip(0, 20) / 20
alpha = 0.02 + 0.05 * speed_conf  # Range: 0.02 to 0.07
fused_speed = (1 - alpha) * v_gps + alpha * v_imu
```

| Flight Phase | GPS Speed | IMU Weight (α) | Reasoning |
|---|---|---|---|
| Hover | ~0 m/s | **2%** | GPS is stable, minimize IMU drift |
| Cruise | ~10 m/s | **4.5%** | Balanced fusion |
| High-speed | 20+ m/s | **7%** | IMU captures rapid dynamics |

#### Vertical Fusion (Adaptive)

```python
# Trust IMU more during aggressive climbs/dives
vertical_accel = az_w.abs()
alpha_z = 0.03 + 0.12 * (vertical_accel.clip(0, 10) / 10)  # Range: 0.03 to 0.15
fused_vz = (1 - alpha_z) * v_gps_z + alpha_z * v_imu_z
```

| Flight Phase | Vertical Accel | IMU Weight (α) | Reasoning |
|---|---|---|---|
| Stable hover | ~0 m/s² | **3%** | GPS altitude stable |
| Gentle climb | ~2 m/s² | **5.4%** | Balanced |
| Aggressive climb | ~5 m/s² | **9%** | IMU responds faster |
| Hard dive | ~10+ m/s² | **15%** | GPS too noisy |

---

## 📊 Key Flight Metrics

| Metric | Description | Calculation Method |
|---|---|---|
| **Max Horizontal Speed** | Peak ground speed (fused GPS+IMU) | `fused_speed.max()` |
| **Max Vertical Speed** | Highest climb/descent rate | `abs(fused_vz).max()` |
| **Max Acceleration** | Peak linear G-force (gravity removed) | `acc_mag.max()` |
| **Max Altitude Gain** | Total cumulative ascent | `delta_alt.clip(≥0).sum()` |
| **Total Distance** | Real 3D path length (Haversine) | `step_dist.sum()` |
| **Flight Duration** | Mission time | `(time_max - time_min) / 1e6` |

---

## 📁 Project Structure

```
Raccoonteam/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── mainapp/
│   └── app.py                   # Streamlit web interface
│
├── parser/
│   ├── __init__.py
│   ├── parser.py                # MAVLink log parser
│   │   ├── TelemetryParser      # Raw .bin → DataFrames
│   │   ├── UnitAutoDetector     # Automatic unit conversion
│   │   └── TelemetryProcessor   # Sensor synchronization
│   ├── data/                    # Input .bin files (gitignored)
│   └── merged_output_*.csv      # Processed output examples
│
├── metrics/
│   ├── __init__.py
│   └── metrics.py               # INS navigation core
│       ├── AHRS orientation
│       ├── Gravity compensation
│       ├── Trapezoidal integration
│       └── Adaptive sensor fusion
│
└── visualization/
    ├── __init__.py
    └── plot.py                  # 3D trajectory visualization
        ├── Geodetic → ENU conversion
        ├── Plotly 3D plots
        └── Mission report overlay
```

---

## ⚙️ Configuration

### Tuning Fusion Parameters

Edit `metrics/metrics.py` to adjust sensor fusion behavior:

```python
# Horizontal fusion
alpha_base = 0.02        # Base IMU trust (lower = more GPS trust)
alpha_dynamic = 0.05     # Additional IMU trust at high speed

# Vertical fusion
alpha_z_base = 0.03      # Base IMU trust for altitude
alpha_z_dynamic = 0.12   # IMU trust scaling with vertical accel

# Smoothing
rolling_window = 5       # Moving average window for fused data
max_step_dist = 50       # Max GPS step (meters) to filter spikes
```

### Parser Settings

The parser automatically handles:
- GPS coordinates in `deg×1e7` or `deg` format
- Altitude in `mm` or `m`
- Acceleration in `milli-g` or `m/s²`
- Gyroscope in `deg/s` or `rad/s`

No manual configuration needed for standard MAVLink logs.

---


## 🧰 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Data Processing** | Pandas, NumPy | DataFrame operations, vectorized math |
| **Mathematics** | SciPy | Quaternions, rotations, transformations |
| **MAVLink** | pymavlink, pymavlog | Binary telemetry log parsing |
| **Geodesy** | haversine, pymap3d | GPS distance, coordinate conversions |
| **Web Framework** | Streamlit | Interactive dashboard |
| **Visualization** | Plotly | 3D trajectory plots |

---

## 📝 License & Credits

**Developed by**: Raccoonteam

**Project**: UAV Telemetry Analytics - Intelligent Navigation Core

**Key Algorithms**:
- AHRS (Attitude and Heading Reference System)
- Complementary Filter Sensor Fusion
- Trapezoidal Numerical Integration
- Haversine Geodesic Distance

*Made with 🚁 by Raccoonteam*
