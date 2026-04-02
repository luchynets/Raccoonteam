import argparse
from pymavlink import mavutil
import pandas as pd
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TelemetryParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.gps_data = []
        self.imu_data = []

    def parse(self):
        logging.info(f"Opening log: {self.filepath}")
        log = mavutil.mavlink_connection(self.filepath)

        while True:
            msg = log.recv_match(blocking=False)
            if msg is None:
                break

            msg_type = msg.get_type()

            try:
                # -------- GPS --------
                if msg_type in ["GPS", "GPS_RAW_INT"]:
                    self.gps_data.append({
                        "time": getattr(msg, "TimeUS", None),
                        "lat": getattr(msg, "Lat", getattr(msg, "lat", None)),
                        "lon": getattr(msg, "Lng", getattr(msg, "lon", None)),
                        "alt": getattr(msg, "Alt", getattr(msg, "alt", None))
                    })

                # -------- IMU --------
                elif msg_type in ["IMU", "RAW_IMU", "SCALED_IMU2"]:
                    self.imu_data.append({
                        "time": getattr(msg, "TimeUS", None),
                        "acc_x": getattr(msg, "AccX", getattr(msg, "xacc", None)),
                        "acc_y": getattr(msg, "AccY", getattr(msg, "yacc", None)),
                        "acc_z": getattr(msg, "AccZ", getattr(msg, "zacc", None)),
                        "gyro_x": getattr(msg, "GyrX", getattr(msg, "xgyro", None)),
                        "gyro_y": getattr(msg, "GyrY", getattr(msg, "ygyro", None)),
                        "gyro_z": getattr(msg, "GyrZ", getattr(msg, "zgyro", None))
                    })

            except Exception as e:
                logging.warning(f"Skipping message due to error: {e}")

        logging.info("Parsing complete")

    def to_dataframe(self):
        gps_df = pd.DataFrame(self.gps_data).dropna()
        imu_df = pd.DataFrame(self.imu_data).dropna()

        return gps_df, imu_df



class TelemetryProcessor:
    @staticmethod
    def normalize_units(gps_df, imu_df):
        logging.info("Normalizing units")

        if not gps_df.empty:
            gps_df["lat"] = gps_df["lat"] / 1e7
            gps_df["lon"] = gps_df["lon"] / 1e7
            gps_df["alt"] = gps_df["alt"] / 1000  # mm -> m

        return gps_df, imu_df

    @staticmethod
    def compute_frequency(df, time_col="time"):
        if df.empty or time_col not in df:
            return 0

        dt = df[time_col].diff().dropna() / 1e6  # µs -> s
        if len(dt) == 0:
            return 0

        return 1 / dt.mean()

    @staticmethod
    def merge(gps_df, imu_df):
        if gps_df.empty or imu_df.empty:
            return None

        logging.info("Merging datasets")
        return pd.merge_asof(
            imu_df.sort_values("time"),
            gps_df.sort_values("time"),
            on="time"
        )

import numpy as np

class UnitAutoDetector:

    # ---------------- GPS ----------------
    @staticmethod
    def detect_gps(df):
        units = {}

        if df.empty:
            return df, units

        # LAT/LON
        if "lat" in df:
            max_val = df["lat"].abs().max()

            if max_val > 1e6:
                # deg * 1e7
                df["lat"] = df["lat"] / 1e7
                units["lat"] = "deg"
            else:
                units["lat"] = "deg"

        if "lon" in df:
            max_val = df["lon"].abs().max()

            if max_val > 1e6:
                df["lon"] = df["lon"] / 1e7
                units["lon"] = "deg"
            else:
                units["lon"] = "deg"

        # ALT
        if "alt" in df:
            max_val = df["alt"].abs().max()

            if max_val > 10000:
                df["alt"] = df["alt"] / 1000
                units["alt"] = "m"
            else:
                units["alt"] = "m"

        return df, units

    # ---------------- IMU ----------------
    @staticmethod
    def detect_imu(df):
        units = {}

        if df.empty:
            return df, units

        # ACCELERATION
        for axis in ["acc_x", "acc_y", "acc_z"]:
            if axis in df:
                max_val = df[axis].abs().max()

                if max_val > 100:  
                    # швидше за все mg або raw
                    df[axis] = df[axis] / 1000  # mg -> g
                    df[axis] = df[axis] * 9.80665
                    units[axis] = "m/s^2"
                else:
                    units[axis] = "m/s^2"

        # GYRO
        for axis in ["gyro_x", "gyro_y", "gyro_z"]:
            if axis in df:
                max_val = df[axis].abs().max()

                if max_val > 50:  
                    # deg/s -> rad/s
                    df[axis] = np.deg2rad(df[axis])
                    units[axis] = "rad/s"
                else:
                    units[axis] = "rad/s"

        return df, units

def main():
    parser = argparse.ArgumentParser(description="Telemetry Parser")
    parser.add_argument("input", help="Path to .bin log file")
    parser.add_argument("--output", default="output.csv", help="Output CSV file")

    args = parser.parse_args()

    # --- Parse ---
    tp = TelemetryParser(args.input)
    tp.parse()

    gps_df, imu_df = tp.to_dataframe()

    gps_df, gps_units = UnitAutoDetector.detect_gps(gps_df)
    imu_df, imu_units = UnitAutoDetector.detect_imu(imu_df)

    print("GPS units:", gps_units)
    print("IMU units:", imu_units)

    gps_freq = TelemetryProcessor.compute_frequency(gps_df)
    imu_freq = TelemetryProcessor.compute_frequency(imu_df)

    logging.info(f"GPS frequency: {gps_freq:.2f} Hz")
    logging.info(f"IMU frequency: {imu_freq:.2f} Hz")

    merged_df = TelemetryProcessor.merge(gps_df, imu_df)

    if merged_df is not None:
        merged_df.to_csv(args.output, index=False)
        logging.info(f"Saved merged data to {args.output}")
    else:
        gps_df.to_csv("gps_" + args.output, index=False)
        imu_df.to_csv("imu_" + args.output, index=False)
        logging.info("Saved separate GPS and IMU files")


if __name__ == "__main__":
    main()

## ЗАПУСК
# python parser.py data_path(e.x.: data/00000001.BIN) --output merged_output.csv
