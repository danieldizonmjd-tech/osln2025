# Starlab Solar Energy Monitor

This project simulates Starlab solar-panel telemetry and Earth solar-farm telemetry, stores every sensor sample in SQLite, runs a local PyTorch anomaly classifier, and shows a live dashboard for power routing, battery charging, light state, and operator-facing impact alerts.

The main cockpit uses three astronaut-facing tabs:

- `Panel Monitoring`: per-array telemetry, anomaly events, sensor mesh health, and routing telemetry
- `Battery`: interactive silicon solid-state battery pack and cell diagram
- `System Energy Flow`: map-style Starlab cutaway showing solar wings, service module, utility bay, Systems Deck, Internal Payload Laboratory, Crew Habitation Deck, upper stowage decks, docking ports, Bishop Airlock, external payload platforms, thermal radiators, propulsion/GNC, and the central power/data spine
- `Benchmarks`: space-mode comparison against the presentation-guide ISS baseline: fixed ~60% battery routing, bulk monitoring, no dedicated panel impact localization, and EVA cost savings estimated at $50k-$100k per EVA hour

## One-command setup

From this folder in PowerShell:

```powershell
.\setup_starlab.ps1
```

The script does the heavy setup:

- installs Python 3.12 through `uv` if needed
- creates `.venv`
- installs app dependencies
- installs CUDA-enabled PyTorch from the official PyTorch CUDA 12.8 wheel index
- creates `starlab_energy.db`
- trains `models/starlab_anomaly_model.pt`

## Run the dashboard

```powershell
.\.venv\Scripts\python.exe predict.py
```

If your terminal is not already in this folder, use the full path:

```powershell
& "C:\Users\Ilyess\Downloads\OSLN-20260507T205707Z-3-001\OSLN\.venv\Scripts\python.exe" "C:\Users\Ilyess\Downloads\OSLN-20260507T205707Z-3-001\OSLN\predict.py"
```

## Database

The app uses `starlab_energy.db` and creates three tables:

- `sensor_readings`: live telemetry from each solar array plus actual and predicted labels
- `sensor_readings.environment_mode`: `space` or `earth`
- `sensor_readings.vibration_g`, `exposure_factor`, `benchmark_power`, and `benchmark_ratio`: mesh-sensor realism used for impact detection and per-panel baselines
- `incidents`: astronaut/operator-facing anomaly and impact alerts
- `system_metrics`: total power, efficiency, battery charge, dynamic charge power, fixed 60% comparison, payload power, local ML latency, environment mode, and CUDA status

The database is initialized automatically by the setup script and on app startup.

## ML model

The model is defined in `ml_model.py` and trained by:

```powershell
.\.venv\Scripts\python.exe train_model.py --samples 52000 --epochs 28 --batch-size 256
```

It trains on mixed space-station and Earth solar-farm simulation data. Space labels include debris impacts, shadowing, dust, overheating, and radiation interference. Earth labels focus on PV operations: cloud cover, partial shading, soiling, benchmark underperformance, string faults, inverter clipping, and overheating.

Space mode is scaled to the presentation guide: Starlab-class 60 kW generation, 12 kW payload target, 45-minute sunlight/shadow intervals, vibration-based impact signatures, and dynamic battery routing compared against the ISS-style fixed 60% charging rule.

It uses CUDA when PyTorch can see an NVIDIA GPU. Verify CUDA with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

PyTorch's official Windows install page recommends choosing the CUDA wheel that matches the system; this setup uses the CUDA 12.8 wheel index because it supports modern NVIDIA drivers and your RTX 5070.
