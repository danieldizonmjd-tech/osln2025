from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


DB_PATH = Path("starlab_energy.db")


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    array_id INTEGER NOT NULL,
    voltage REAL NOT NULL,
    current REAL NOT NULL,
    power REAL NOT NULL,
    temperature REAL NOT NULL,
    irradiance REAL NOT NULL,
    panel_angle REAL NOT NULL,
    efficiency REAL NOT NULL,
    radiation_level REAL NOT NULL,
    surface_condition REAL NOT NULL,
    battery_charge REAL NOT NULL,
    sunlight INTEGER NOT NULL,
    actual_label TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    environment_mode TEXT NOT NULL DEFAULT 'space',
    vibration_g REAL NOT NULL DEFAULT 0,
    exposure_factor REAL NOT NULL DEFAULT 1,
    benchmark_power REAL NOT NULL DEFAULT 0,
    benchmark_ratio REAL NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    array_id INTEGER NOT NULL,
    incident_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    message TEXT NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_power REAL NOT NULL,
    average_efficiency REAL NOT NULL,
    total_energy_wh REAL NOT NULL,
    battery_charge REAL NOT NULL,
    power_mode TEXT NOT NULL,
    excess_power REAL NOT NULL,
    cuda_enabled INTEGER NOT NULL,
    environment_mode TEXT NOT NULL DEFAULT 'space',
    fixed_charge_power REAL NOT NULL DEFAULT 0,
    dynamic_charge_power REAL NOT NULL DEFAULT 0,
    battery_priority_fraction REAL NOT NULL DEFAULT 0,
    payload_power REAL NOT NULL DEFAULT 0,
    science_headroom REAL NOT NULL DEFAULT 0,
    power_margin REAL NOT NULL DEFAULT 0,
    battery_signed_power REAL NOT NULL DEFAULT 0,
    battery_pack_voltage REAL NOT NULL DEFAULT 0,
    battery_pack_current REAL NOT NULL DEFAULT 0,
    battery_pack_temp REAL NOT NULL DEFAULT 0,
    local_latency_ms REAL NOT NULL DEFAULT 0
);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def init_database(db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)
        _ensure_column(connection, "sensor_readings", "environment_mode", "TEXT NOT NULL DEFAULT 'space'")
        _ensure_column(connection, "sensor_readings", "vibration_g", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "sensor_readings", "exposure_factor", "REAL NOT NULL DEFAULT 1")
        _ensure_column(connection, "sensor_readings", "benchmark_power", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "sensor_readings", "benchmark_ratio", "REAL NOT NULL DEFAULT 1")
        _ensure_column(connection, "system_metrics", "environment_mode", "TEXT NOT NULL DEFAULT 'space'")
        _ensure_column(connection, "system_metrics", "fixed_charge_power", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "dynamic_charge_power", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "battery_priority_fraction", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "payload_power", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "science_headroom", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "power_margin", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "battery_signed_power", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "battery_pack_voltage", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "battery_pack_current", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "battery_pack_temp", "REAL NOT NULL DEFAULT 0")
        _ensure_column(connection, "system_metrics", "local_latency_ms", "REAL NOT NULL DEFAULT 0")


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def insert_readings(rows: Iterable[dict], db_path: Path | str = DB_PATH) -> None:
    rows = list(rows)
    if not rows:
        return
    with connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO sensor_readings (
                timestamp, array_id, voltage, current, power, temperature,
                irradiance, panel_angle, efficiency, radiation_level,
                surface_condition, battery_charge, sunlight, actual_label,
                predicted_label, confidence, environment_mode, vibration_g,
                exposure_factor, benchmark_power, benchmark_ratio
            ) VALUES (
                :timestamp, :array_id, :voltage, :current, :power, :temperature,
                :irradiance, :panel_angle, :efficiency, :radiation_level,
                :surface_condition, :battery_charge, :sunlight, :actual_label,
                :predicted_label, :confidence, :environment_mode, :vibration_g,
                :exposure_factor, :benchmark_power, :benchmark_ratio
            )
            """,
            rows,
        )


def insert_metric(metric: dict, db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO system_metrics (
                timestamp, total_power, average_efficiency, total_energy_wh,
                battery_charge, power_mode, excess_power, cuda_enabled,
                environment_mode, fixed_charge_power, dynamic_charge_power,
                battery_priority_fraction, payload_power, science_headroom,
                power_margin, battery_signed_power, battery_pack_voltage,
                battery_pack_current, battery_pack_temp, local_latency_ms
            ) VALUES (
                :timestamp, :total_power, :average_efficiency, :total_energy_wh,
                :battery_charge, :power_mode, :excess_power, :cuda_enabled,
                :environment_mode, :fixed_charge_power, :dynamic_charge_power,
                :battery_priority_fraction, :payload_power, :science_headroom,
                :power_margin, :battery_signed_power, :battery_pack_voltage,
                :battery_pack_current, :battery_pack_temp, :local_latency_ms
            )
            """,
            metric,
        )


def insert_incident(incident: dict, db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO incidents (
                timestamp, array_id, incident_type, severity, confidence, message
            ) VALUES (
                :timestamp, :array_id, :incident_type, :severity, :confidence, :message
            )
            """,
            incident,
        )


def acknowledge_all(db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.execute("UPDATE incidents SET acknowledged = 1 WHERE acknowledged = 0")


def recent_readings(limit: int = 64, db_path: Path | str = DB_PATH) -> list[sqlite3.Row]:
    with connect(db_path) as connection:
        return list(
            connection.execute(
                "SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ?", (limit,)
            )
        )


def recent_incidents(limit: int = 20, db_path: Path | str = DB_PATH) -> list[sqlite3.Row]:
    with connect(db_path) as connection:
        return list(
            connection.execute(
                "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
            )
        )


def database_counts(db_path: Path | str = DB_PATH) -> dict[str, int]:
    with connect(db_path) as connection:
        tables = ("sensor_readings", "incidents", "system_metrics")
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
