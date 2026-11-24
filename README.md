# OSLN 2025

This repository contains a Starlab solar-energy simulator and an Arduino light-monitoring tool.

## Solaris

[solaris-main](solaris-main/) simulates solar-panel telemetry, stores sensor readings in SQLite, and displays power routing, battery status, and anomaly alerts in a live dashboard. See the [Solaris setup guide](solaris-main/SETUP.md) for installation and launch instructions.

## photoneye

[photoneye](photoneye/) reads light levels from an Arduino and plots them live. The [Arduino sketch](photoneye/arduino.cpp) reads analog pin A0 and sends values over serial at 115200 baud. The [Python monitor](photoneye/realtime.py) logs readings to CSV and saves a run summary when the plot window closes.

1. Upload the sketch to the Arduino with the light sensor connected to A0.
2. In `photoneye/realtime.py`, set `arduino_port` and `output_dir` for your machine.
3. From the repository root, run:

   ```sh
   python -m pip install matplotlib numpy pytz pyserial
   python photoneye/realtime.py
   ```
