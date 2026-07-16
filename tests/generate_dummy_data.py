"""Generate a small longitudinal PPMI-shaped dataset for local testing.

Run from the repository root with:
    python tests/generate_dummy_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_PATIENTS = 50
VISITS = (("BL", 0), ("V01", 180), ("V02", 365))


def generate_dummy_data(output_path: str | Path | None = None) -> Path:
    """Create a reproducible, clinically plausible longitudinal CSV fixture."""
    rng = np.random.default_rng(RANDOM_SEED)
    output = Path(output_path) if output_path else Path(__file__).with_name("dummy_ppmi.csv")
    records = []

    for patno in range(100001, 100001 + N_PATIENTS):
        sex = int(rng.integers(1, 3))
        age = int(rng.integers(45, 81))
        education = int(rng.integers(8, 22))
        baseline_duration = round(float(rng.uniform(0.0, 8.0)), 1)
        baseline_upsit = int(rng.integers(12, 37))
        baseline_moca = int(rng.integers(20, 31))
        baseline_updrs = int(rng.integers(5, 46))
        baseline_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))

        for visit_index, (event_id, days_from_baseline) in enumerate(VISITS):
            moca = int(np.clip(baseline_moca - visit_index * rng.uniform(0.2, 1.8) + rng.normal(0, 1.0), 0, 30))
            updrs = int(np.clip(baseline_updrs + visit_index * rng.uniform(2.0, 8.0) + rng.normal(0, 3.0), 0, 100))
            visit_age = age + days_from_baseline / 365.25
            duration = round(baseline_duration + days_from_baseline / 365.25, 2)
            upsit = int(np.clip(baseline_upsit - visit_index * rng.uniform(0.0, 1.2) + rng.normal(0, 1.0), 0, 40))
            visit_date = baseline_date + pd.Timedelta(days=days_from_baseline)

            records.append(
                {
                    "PATNO": patno,
                    "EVENT_ID": event_id,
                    "MOCA": moca,
                    "UPDRS_III": updrs,
                    "AGE": round(visit_age, 2),
                    "SEX": sex,
                    "EDUCYRS": education,
                    "DURATION": duration,
                    "UPSIT": upsit,
                    "visit_date": visit_date.date().isoformat(),
                    # Canonical names consumed by the PPMI pipeline.
                    "moca": moca,
                    "updrs3_score": updrs,
                    "age": round(visit_age, 2),
                    "duration": duration,
                    "upsit": upsit,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    csv_path = generate_dummy_data()
    print(f"Created {csv_path} with {N_PATIENTS} patients and {N_PATIENTS * len(VISITS)} visits.")
