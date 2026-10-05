"""
generate_data.py
-----------------
Generates a synthetic time-series dataset that simulates sensor readings
from marine diesel engines / propulsion systems, for use in a predictive
maintenance (PdM) machine learning pipeline.

Each row represents one operating "cycle" (e.g. one hour of operation) for
a given engine unit. As an engine approaches failure, several sensor
channels drift away from their healthy baseline (a common pattern in
real turbofan/engine degradation datasets such as NASA C-MAPSS, which
this generator is loosely inspired by).

Output: data/maritime_sensor_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

N_UNITS = 40          # number of vessels/engines simulated
MIN_LIFE = 120         # minimum cycles before failure
MAX_LIFE = 380         # maximum cycles before failure
FAILURE_WINDOW = 30    # cycles before failure considered "high risk" (label=1)


def simulate_unit(unit_id: int) -> pd.DataFrame:
    life = RNG.integers(MIN_LIFE, MAX_LIFE)
    cycles = np.arange(1, life + 1)
    frac = cycles / life  # 0 -> healthy, 1 -> failure point

    # Baseline healthy operating values (typical marine diesel engine ranges)
    exhaust_temp = 380 + 5 * RNG.standard_normal(life)          # deg C
    coolant_temp = 75 + 2 * RNG.standard_normal(life)           # deg C
    lube_oil_pressure = 4.2 + 0.15 * RNG.standard_normal(life)  # bar
    vibration_rms = 2.0 + 0.2 * RNG.standard_normal(life)       # mm/s
    rpm = 720 + 10 * RNG.standard_normal(life)                  # rpm
    fuel_rate = 180 + 6 * RNG.standard_normal(life)             # kg/h
    oil_particle_count = 8 + 1.5 * RNG.standard_normal(life)    # ISO code proxy
    hull_speed = 14 + 0.5 * RNG.standard_normal(life)           # knots

    # Degradation trends: nonlinear drift that accelerates near end-of-life
    degradation = frac ** 2.5
    exhaust_temp += degradation * 90
    coolant_temp += degradation * 18
    lube_oil_pressure -= degradation * 1.6
    vibration_rms += degradation * 5.5
    oil_particle_count += degradation * 14
    fuel_rate += degradation * 25

    # remaining useful life (RUL), capped is not necessary here
    rul = life - cycles

    # binary label: 1 if within FAILURE_WINDOW cycles of failure
    label = (rul <= FAILURE_WINDOW).astype(int)

    df = pd.DataFrame({
        "unit_id": unit_id,
        "cycle": cycles,
        "exhaust_temp_C": exhaust_temp,
        "coolant_temp_C": coolant_temp,
        "lube_oil_pressure_bar": lube_oil_pressure,
        "vibration_rms_mm_s": vibration_rms,
        "rpm": rpm,
        "fuel_rate_kg_h": fuel_rate,
        "oil_particle_count": oil_particle_count,
        "hull_speed_knots": hull_speed,
        "RUL": rul,
        "failure_within_30cy": label,
    })
    return df


def main():
    frames = [simulate_unit(u) for u in range(1, N_UNITS + 1)]
    data = pd.concat(frames, ignore_index=True)

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "maritime_sensor_data.csv"
    data.to_csv(out_path, index=False)

    print(f"Generated {len(data)} rows across {N_UNITS} engine units -> {out_path}")

    print()
    print("First 5 rows:")
    print(data.head())

    print()
    print("Dataset shape:")
    print(data.shape)

    print()
    print("Columns:")
    print(data.columns)

    print()
    print("Maximum cycle for each engine:")
    print(data.groupby("unit_id")["cycle"].max())

    print()
    print("Label distribution:")
    print(data["failure_within_30cy"].value_counts(normalize=True))

if __name__ == "__main__":
    main()