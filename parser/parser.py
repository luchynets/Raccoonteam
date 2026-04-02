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


def main():
    parser = argparse.ArgumentParser(description="Telemetry Parser")
    parser.add_argument("input", help="Path to .bin log file")
    parser.add_argument("--output", default="output.csv", help="Output CSV file")

    args = parser.parse_args()

    # --- Parse ---
    tp = TelemetryParser(args.input)
    tp.parse()

    gps_df, imu_df = tp.to_dataframe()

    gps_df, imu_df = TelemetryProcessor.normalize_units(gps_df, imu_df)

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
