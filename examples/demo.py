"""Demo: daily runs of the virtual-sensor transform.

    python examples/demo.py

Reads 15 days of MKP data (``mkp_sample_2024-06-01_15.csv``: 14 days of
spin-up + the report day 2024-06-15) and

1. predicts virtual twins of real MKP sensors and compares them with the
   measurements (explicit ``virtual_sensors`` in model coordinates);
2. runs exactly what the platform would call, ``transform(add_wind_speed(df))`` with only
   the MKP channels: the virtual sensors come from MKP's sensor IFC model
   (packaged mapping of ``E+233TUS_*``) and the wind speed from Open-Meteo.
   The result is written next to this script as ``demo_output.csv``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nibelungen_virtual_sensors import DEFAULT_INPUT_CHANNELS, add_wind_speed, transform

HERE = Path(__file__).resolve().parent

# Virtual twins of measured MKP channels (model coordinates, see README):
# pilot section E+4.0 m = model z 0 (calibrated positions) and the
# inclinometer at E+42.3 m = model z 38.3.
TWINS = {
    "E_plus_040TU_HS--u-_Avg1": {"quantity": "temperature", "position": [2.85, -5.264, 0.0], "noise_std": 0.1468},
    "E_plus_040TU_HS--o-_Avg1": {"quantity": "temperature", "position": [2.85, -0.171, 0.0], "noise_std": 0.3152},
    "E_plus_040TU_HSS-m-_Avg1": {"quantity": "temperature", "position": [3.450, -2.63, 0.0], "noise_std": 0.4388},
    "E_plus_040TU_HSN-m-_Avg1": {"quantity": "temperature", "position": [2.081, -2.63, 0.0], "noise_std": 0.2168},
    "E_plus_423NU_HSN-o-_Avg1": {"quantity": "inclination", "position": [-2.25, -0.2, 38.3]},
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = pd.read_csv(HERE / "mkp_sample_2024-06-01_15.csv", parse_dates=["Timestamp"])
    raw["Timestamp"] = raw["Timestamp"].astype("datetime64[ns]")
    df = raw[["Timestamp", *DEFAULT_INPUT_CHANNELS.values()]]     # what the platform passes

    sensors = [{"name": f"VS_{k}", **v} for k, v in TWINS.items()]
    t0 = time.perf_counter()
    out = transform(df, sensors)
    print(f"\ntransform: {len(out)} rows x {out.shape[1]} columns in {time.perf_counter() - t0:.1f} s")
    print(out.iloc[::8, :5].to_string(index=False))

    meas = raw.set_index("Timestamp").loc[out["Timestamp"]]
    print("\nvirtual twin vs. measurement on", out["Timestamp"].dt.date.iloc[0])
    for k, spec in TWINS.items():
        y, m = meas[k].to_numpy(), out[f"VS_{k}"].to_numpy()
        unit = out.attrs["units"][f"VS_{k}"]
        line = f"  {k:26s} corr {np.corrcoef(y, m)[0, 1]:.2f}"
        if spec["quantity"] == "inclination":
            # the inclinometer zero is arbitrary: compare after removing the mean offset
            r = (y - y.mean()) - (m - m.mean())
            line += f"  RMSE after offset {np.sqrt(np.mean(r ** 2)):.3f} {unit}"
        else:
            inside = np.mean((y >= out[f"VS_{k}_lower"]) & (y <= out[f"VS_{k}_upper"]))
            line += f"  RMSE {np.sqrt(np.mean((y - m) ** 2)):.3f} {unit}  in 95% band {inside:.0%}"
        print(line)

    # 2. the platform call: MKP channels only
    mkp_only = df.drop(columns=[DEFAULT_INPUT_CHANNELS["wind_speed"]])
    try:
        out = transform(add_wind_speed(mkp_only))       # wind from Open-Meteo
    except RuntimeError as exc:                         # offline
        print(f"\n{exc}\n-> using the wind column of the sample instead")
        out = transform(df)
    print("\nIFC virtual sensors (E+233TUS_*), daily min / max of the predictive mean:")
    for name, unit in out.attrs["units"].items():
        print(f"  {name:18s} {out[name].min():6.2f} / {out[name].max():6.2f} {unit}"
              f"   mean 95% half-width {(out[name + '_upper'] - out[name]).mean():.2f} {unit}")
    out.to_csv(HERE / "demo_output.csv", index=False)
    print(f"\nwritten {HERE / 'demo_output.csv'}")


if __name__ == "__main__":
    main()
