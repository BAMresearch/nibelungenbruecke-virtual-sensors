# nibelungen-virtual-sensors

Virtual sensors for the digital twin of the Nibelungen bridge in Worms, developed at BAM.

A calibrated, stochastic thermomechanical model of the instrumented cantilever ("Kragarm E-Ost") predicts temperatures, tilt, displacements and strains at points where nothing is measured. Every prediction comes with a credible interval. The entry point follows the platform signature agreed with MKP:

```python
transform(df: pd.DataFrame, param1, param2, ...) -> pd.DataFrame
```

## Installation (environment)

```bash
pip install .            # numpy, scipy, pandas
pip install ".[ifc]"     # + ifcopenshell, only to read virtual sensors from another IFC file at run time
pip install ".[test]"    # + pytest
```

All dependencies are in `pyproject.toml`.
- **Python:** 3.9 or newer.
- **Libraries:** pure Python wheels only.
- **Tested:** Python 3.10 (NumPy 1.26, SciPy 1.10, pandas 2.3) and Python 3.12 (NumPy 2.5, SciPy 1.18, pandas 3.0).
- **Files:** the package reads only files shipped inside it (mesh, calibrated parameters, sensor definitions), through `importlib.resources` and `pathlib`.

## Usage

```python
from nibelungen_virtual_sensors import add_wind_speed, transform

result = transform(add_wind_speed(df))   
```

With only the MKP channels in `df`:
- the virtual sensors are the four temperature sensors `E+233TUS_*`;
- `add_wind_speed(df)` attaches the wind speed from Open-Meteo.

Optional variants of the call:

```python
df.attrs.update(report_start="2024-06-15T00:00", report_end="2024-06-15T23:30")
result = transform(df)
result = transform(df, sensor_ifc="path/to/sensors.ifc")      # other IFC model, needs [ifc]
result = transform(df, virtual_sensors=[                      # explicit points, model coordinates
    {"name": "VS_T_deck", "quantity": "temperature", "position": [2.85, -0.05, 19.3], "position_std": 0.02},
    {"name": "VS_tilt",   "quantity": "inclination", "position": [-2.25, -0.2, 38.3]},
])
result = transform(df, position_uncertainty=False)            # nominal sensor positions only (see below)
```

### Input `df`

Schema: `Timestamp: datetime64[ns]` (UTC), then one `float64` column per channel.
- Any sampling rate works; the data are bin-averaged to the 30-min model step.
- Gaps are interpolated.
- Other columns are ignored.

| Model input | Channel | Unit |
|---|---|---|
| Outside air temperature | `F_plus_000TA_KaS-o-_Avg1` | °C |
| Air temperature inside the box girder | `E_plus_040TI_HSS-u-_Avg` | °C |
| Global shortwave irradiation | `F_plus_000S_KaS-o-_Avg1` | W/m² |
| Wind speed (optional, see below) | `wind_speed_10m` | m/s |

The channel names are those of the MKP data API. Pass `input_channels={"air_temperature": "...", ...}` if the platform uses other names.

**Wind speed** is not measured by MKP. `transform` needs the wind column, which the separate step `add_wind_speed(df)` attaches (no model, so it can run in its own container). By default it fetches hourly `wind_speed_10m` from api.open-meteo.com over HTTPS:
- **Grid point:** 49.9 N, 8.35 E, the same as the calibration.
- **Endpoint:** the archive endpoint for dates more than 6 days back, the forecast endpoint for recent days.
- **Without network:** `add_wind_speed(df, 2.7)` attaches a constant in m/s, or add the column yourself.

### Parameters (all basic Python types, only df is compulsory)

| Parameter | Default | Meaning |
|---|---|---|
| `virtual_sensors` | the 4 packaged IFC sensors | `list` of `dict`: `name`, `quantity`, `position` `[x, y, z]` (model coordinates, m), optional `noise_std` (residual error added to the variance, default 0: only the embedded MFU is propagated); optional for temperature sensors `position_std` or `member` + `depth_range` (position uncertainty, see below) |
| `sensor_ifc` | `None` | path to an IFC file whose sensor proxies define the virtual sensors |
| `df.attrs["report_start"]`, `df.attrs["report_end"]` | the last 24 h of `df` | ISO-8601 limits, both inclusive, of the returned rows (properties of `df`, not arguments) |
| `timestep` | `"30min"` | model time step (the calibration step) |
| `input_channels` | table above | `dict` role → column name |
| `confidence_level` | `0.95` | level of the `_lower` / `_upper` columns |
| `initial_condition_steps` | `100` | steps repeating the first input row before the history |
| `min_history_hours` | `336` | below this history length a warning is logged |
| `model_overrides` | `None` | custom model properties as nested `dict` merged into `data/model_config.json` |
| `position_uncertainty` | `True` | include the sensor-position uncertainty in the distribution of the temperature sensors (see "Sensor-position uncertainty") |

### Output

`Timestamp: datetime64[ns]`, one row per 30 min of the report window, then four `float64` columns per virtual sensor `<name>`:

| Column | Content |
|---|---|
| `<name>` | predicted value (predictive mean) |
| `<name>_std` | predictive standard deviation (for temperature sensors including the position uncertainty) |
| `<name>_lower`, `<name>_upper` | lower and upper limit of the credible interval at `confidence_level` |

`result.attrs` holds metadata for the dashboard: `units` per sensor, `confidence_level`, the spin-up length, the backend and `position_uncertainty` (whether the term was applied, to which sensors, number of samples).

With the default sensors, `transform(df)` returns 17 columns: `Timestamp` and, for each of `E+233TUS_HS--o-`, `E+233TUS_HS--u-`, `E+233TUS_HSS-m-` and `E+233TUS_HSN-m-`, the mean, `_std`, `_lower` and `_upper` columns. All values are in °C.

| quantity | unit | meaning |
|---|---|---|
| `temperature` | °C | concrete temperature |
| `displacement_x`, `displacement_y`, `displacement_z` | mm | displacement; y is vertical, positive upwards |
| `inclination` | mm/m | tilt ∂u_y/∂z |
| `strain_xx` … `strain_xz` | µm/m | total strain tensor |

The tilt is relative to a uniform 20 °C state. A real inclinometer has an arbitrary zero, so compare changes, not levels.


## Demo and tests

```bash
python examples/demo.py      # one daily run on real MKP data (2024-06-01 … 06-15)
pytest                       # ~3 min, peak 1.6 GB (the P2 direct-solver checks); IFC/network tests skip without ifcopenshell/network
```

The demo does two things:
1. It predicts "virtual twins" of measured MKP sensors and compares them with the measurements.
2. It runs the platform call `transform(df)` and writes `examples/demo_output.csv`.

The example data `examples/mkp_sample_2024-06-01_15.csv` uses the MKP channel names.

The notebook `examples/daily_digital_twin_run.ipynb` replays the operational setting: the daily container job queries the data API, calls `transform`, checks the result, writes it back for the dashboard and fails cleanly on missing data. It ends with templates for the image and the schedule.

Results on the report day:

| Twin | Correlation with measurement | Error | Inside 95 % band |
|---|---|---|---|
| Bottom temperature | not informative (daily signal ±0.1 °C, below the sensor noise) | RMSE 0.21 °C | 100 % |
| Top temperature | 0.96 | RMSE 0.61 °C | 100 % |
| South web temperature | 0.69 | RMSE 0.42 °C | 100 % |
| North web temperature | 0.98 | RMSE 0.12 °C | 100 % |
| Tilt | 0.98 | RMSE 0.15 mm/m after removing the zero offset | — |

The twins are explicit sensors at their calibrated positions, so they carry no position term.

## Model (background)

- **Geometry.** 3-D model of the cantilever from the instrumented section E+4.0 m (clamped) to the hinge:
  - 7,036 vertices and 21,737 tetrahedra (`data/mesh_3d.npz`, 0.4 MB), with **quadratic (P2) elements**: 42,848 temperature unknowns;
  - the mesh is the research pipeline's `models/mesh_3d_thermal.msh` (the model ends at the hinge, z = 40.1 m; fitted taper), exported with its boundary tags;
  - two box girders;
  - the depth tapers from 5.87 m (bottom slab 0.87 m, Fig. 2b of the paper) to 2.50 m.
- **Thermal model.** Transient heat conduction (implicit Euler, 30 min), with three boundary conditions:
  - wind-forced convection on the outer surfaces;
  - absorbed shortwave radiation on the deck;
  - convection to the measured air temperature inside the box girder.
- **Mechanical model.** Linear thermo-elasticity of the clamped cantilever under the temperature field: E = 34 GPa, ν = 0.2, α_T = 8·10⁻⁶ 1/K, same P2 elements.
  - A mechanical reading is a linear function of the temperature. One adjoint solve per mechanical sensor, done once per call, therefore replaces a solve per time step and per quadrature node.
  - The solver is ILU-preconditioned GMRES (30–50 iterations), so no factorisation of the 129,000-unknown stiffness matrix is needed (peak memory 1.0 GB).
- **Element degree.** P2 is the default, as in the calibrated 2-D model. The study in `MODEL_ERRORS_GEOMETRY_DISCRETIZATION.md` of the research repository found that linear (P1) elements leave about 0.3 °C and 20 % (tilt, tip deflection) error at any mesh size, against < 0.1 °C and 1 % for P2. P1 stays available for quick checks: `model_overrides={"thermal": {"degree": 1}, "mechanical": {"degree": 1}}`.
- **Calibration.** Updated 2-D calibration of the cross-section (`references/Table_01_hr_vs_no_hr_inference_table_no_hr.tex`, "Final"; 0.87 m bottom slab, ν_air = 1.5·10⁻⁵ m²/s); values in `data/model_config.json`.
  - The wind-convection constant c_c is uncertain: LogNormal with mean 1.0502 and standard deviation 0.7763 (c_r = 1.0163 for the shortwave constant).
  - The model runs at the 3 Gauss nodes of this distribution, which is equivalent to an order-2 polynomial chaos.
  - The interval contains only what the calibration propagates: the uncertainty of c_c and, for temperatures, the sensor position. The residual error calibrated at the pilot sensors (0.15–0.44 °C), the part the model does not explain, is not transferred to virtual sensors.
- **Validation.**
  - With P2 on the packaged mesh, this package reproduces the stored results of the FEniCSx research code (June 2024 thermal ensemble, June 2025 hindcast) to 7.1e-9 K for temperatures and 1.7e-10 (relative) for tilt and tip displacement, rerun with the updated calibration (`tools/validate_against_pipeline.py`, 11 min, 0.9 GB).
  - Against measurements on the demo day (June 2024), the 95 % intervals of the four pilot sensors contain 100 % of the measured temperatures (table above).
  - A systematic seasonal validation of the 3-D model is ongoing at BAM.
- **Known limitations.**
  - Winter predictions are biased (top of the deck +1.7 °C) and their intervals are too narrow.
  - The asphalt layer is not modelled, so the measured tilt lags the model by about 1 h.
  - The model has no lateral solar exposure.
  - The position term is a prior, not a calibration; see "Sensor-position uncertainty".

## Model coordinates

All positions are in metres:

- `x` is transverse. **The southern box girder is at `x > 0`.** Its cavity spans x ∈ [2.25, 3.45]; its north web is x ∈ [1.85, 2.25] and its south web x ∈ [3.45, 3.85].
- `y` is vertical, positive up. `y = 0` is the deck top; the deck is 0.25 m thick.
- `z` runs along the span from the clamped section (`z = 0`) to the hinge, where the model ends (`z = 40.1`). **z = station after pier axis E − 4.0 m**; for example, measuring axis `E+233` is at z = 19.3.

Points must lie inside the concrete; a position in the cavity or in the air raises `ValueError`.

### Aligning the coordinate systems

A plain rigid transformation would get several of these points wrong:

1. **World frame.**
   - The IFC models use ETRS89/UTM 32 with the zone prefix (E ≈ 32 455 100 m) and NHN heights.
   - None of them has an `IfcMapConversion`; one would be applied if present.
2. **Bridge frame.**
   - The plan axis is the principal direction of the superstructure, at a bearing of 73.9°.
   - Station 0 is the centre of the pier head at axis E.
   - The station of the IFC sensor geometry (23.30 m) matches the measuring axis `NiE+233`.
3. **Station → model `z`.** The model's `z = 0` section is the calibrated pilot section E+4.0 m, not the pier axis.
   - With `z = station − 4.0 m`, the IFC and mesh girder depths agree to 0.11 m RMS; with `z = station` the error is 0.31 m.
   - The anchor is `frame.model_z0_station` in `data/ifc_config.json`.
4. **Mirror.** The model is the mirror image of the real bridge, so **model `x` = −(horizontal offset towards the north)**.
   - A rotation alone cannot produce this.
   - The predictions are symmetric in `x`, but the positions must still be mirrored.
5. **Vertical.** The real deck has a gradient and a cross-fall that the model lacks. Heights are therefore never copied: points are placed relative to the faces of the structural member.
6. **Different cross-sections.**

   | at E+23.3 | IFC | Model (z = 19.3) |
   |---|---|---|
   | Deck thickness | 0.28 m | 0.25 m |
   | Cell width | 1.30 m | 1.20 m |
   | Bottom slab thickness | 0.39 m | 0.53 m |
   | Cavity height | 2.93 m | 2.92 m |

   The IFC sensor solids are 25 cm marker cubes. Their centres lie only 7.5 cm deep in the members, and the bottom one lies in the cavity, although the property text says "20 cm (10 cm) tief von der Innenseite".

Each sensor is therefore rebuilt in the model from its description relative to its box-girder cell:

| Item | Primary source | Fallback |
|---|---|---|
| Station | measuring axis (`NiE+233`) | IFC geometry |
| Member (deck, bottom slab, south/north web) | `ePset_Position` text | position relative to the IFC cell |
| Depth behind the cavity face | text ("20 cm tief von der Innenseite") | measured in the IFC section |
| Position along the face | text ("zentriert"/"mittig" → centre) | measured in the IFC section |

- **No residual error.** IFC sensors carry only the propagated embedded MFU and the position term. The residual error calibrated at the pilot sensors is not transferred; pass `noise_std` explicitly to add one.
- **Disagreements.** Where the text and the geometry disagree, this is logged and recorded in the sensor's `mapping` block.
- **Supported sensors.** Only temperature (`°C`) and inclination (`mm/m`) sensors are recognised so far.

Result for the TUS model, all at z = 19.3 m:

| Channel | Member | Model position [x, y] |
|---|---|---|
| `E+233TUS_HS--o-` | deck, 20 cm behind the cavity face | [2.85, −0.050] |
| `E+233TUS_HS--u-` | bottom slab, 10 cm | [2.85, −3.284] |
| `E+233TUS_HSS-m-` | south web, 20 cm | [3.65, −1.717] |
| `E+233TUS_HSN-m-` | north web, 20 cm | [2.05, −1.717] |

The y of the bottom slab and the webs depends on the taper of the mesh: with the mesh of the former, longer model they were −3.225 and −1.688.
