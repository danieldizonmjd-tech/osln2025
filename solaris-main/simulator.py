from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from datetime import datetime


ENVIRONMENT_MODES = ("space", "earth")
STARLAB_RATED_POWER_W = 60_000.0
STARLAB_PAYLOAD_POWER_W = 12_000.0
STARLAB_BASE_LOAD_W = 32_000.0
ISS_FIXED_BATTERY_FRACTION = 0.60

FEATURE_NAMES = [
    "environment_code",
    "cloud_cover",
    "benchmark_ratio",
    "vibration_g",
    "exposure_factor",
    "voltage",
    "current",
    "power",
    "temperature",
    "irradiance",
    "panel_angle",
    "efficiency",
    "radiation_level",
    "surface_condition",
    "storage_voltage",
    "storage_current",
    "degradation",
    "time_since_event",
]

ANOMALY_CLASSES = [
    "NORMAL",
    "SMALL_IMPACT",
    "LARGE_IMPACT",
    "PARTIAL_SHADOW",
    "DUST",
    "OVERHEAT",
    "RADIATION",
    "CLOUD_COVER",
    "SOILING",
    "BENCHMARK_UNDERPERFORMANCE",
    "STRING_FAULT",
    "INVERTER_CLIPPING",
]

IMPACT_CLASSES = {"SMALL_IMPACT", "LARGE_IMPACT"}
SPACE_EVENT_PROBABILITIES = {
    # Per array per live reading. Critical damage should be mission-rare in
    # normal mode; manual injection and test mode still exist for demos.
    "SMALL_IMPACT": 0.000006,
    "LARGE_IMPACT": 0.00000008,
    "PARTIAL_SHADOW": 0.0045,
    "DUST": 0.0012,
    "OVERHEAT": 0.0008,
    "RADIATION": 0.0007,
}
SPACE_TEST_EVENT_PROBABILITIES = {
    "SMALL_IMPACT": 0.0025,
    "LARGE_IMPACT": 0.00045,
    "PARTIAL_SHADOW": 0.03,
    "DUST": 0.015,
    "OVERHEAT": 0.01,
    "RADIATION": 0.009,
}
SPACE_ANOMALY_CLASSES = [
    "NORMAL",
    "SMALL_IMPACT",
    "LARGE_IMPACT",
    "PARTIAL_SHADOW",
    "DUST",
    "OVERHEAT",
    "RADIATION",
]
EARTH_ANOMALY_CLASSES = [
    "NORMAL",
    "PARTIAL_SHADOW",
    "CLOUD_COVER",
    "SOILING",
    "BENCHMARK_UNDERPERFORMANCE",
    "STRING_FAULT",
    "INVERTER_CLIPPING",
    "OVERHEAT",
]


@dataclass
class SolarPanelReading:
    array_id: int
    timestamp: str
    voltage: float
    current: float
    power: float
    temperature: float
    irradiance: float
    panel_angle: float
    efficiency: float
    radiation_level: float
    surface_condition: float
    storage_voltage: float
    storage_current: float
    degradation: float
    time_since_event: float
    sunlight: bool
    environment_mode: str
    vibration_g: float = 0.0
    exposure_factor: float = 1.0
    cloud_cover: float = 0.0
    benchmark_power: float = 0.0
    actual_label: str = "NORMAL"

    @property
    def environment_code(self) -> float:
        return 1.0 if self.environment_mode == "earth" else 0.0

    @property
    def benchmark_ratio(self) -> float:
        if self.benchmark_power <= 1:
            return 1.0
        return max(0.0, min(1.8, self.power / self.benchmark_power))

    def features(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def as_database_row(self, predicted_label: str, confidence: float, battery_charge: float) -> dict:
        return {
            "timestamp": self.timestamp,
            "array_id": self.array_id,
            "voltage": self.voltage,
            "current": self.current,
            "power": self.power,
            "temperature": self.temperature,
            "irradiance": self.irradiance,
            "panel_angle": self.panel_angle,
            "efficiency": self.efficiency,
            "radiation_level": self.radiation_level,
            "surface_condition": self.surface_condition,
            "battery_charge": battery_charge,
            "sunlight": 1 if self.sunlight else 0,
            "actual_label": self.actual_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "environment_mode": self.environment_mode,
            "vibration_g": self.vibration_g,
            "exposure_factor": self.exposure_factor,
            "benchmark_power": self.benchmark_power,
            "benchmark_ratio": self.benchmark_ratio,
        }


class SolarArraySimulator:
    def __init__(self, array_count: int = 4) -> None:
        self.array_count = array_count
        self.start_time = time.time()
        self.sunlight_duration = 45 * 60
        self.shadow_duration = 45 * 60
        self.manual_sunlight: bool | None = None
        self.battery_charge = 72.0
        self.total_energy_wh = 0.0
        self.surface_condition = [100.0 for _ in range(array_count)]
        self.degradation = [0.0 for _ in range(array_count)]
        self.time_since_event = [900.0 for _ in range(array_count)]
        self.forced_events: dict[int, str] = {}
        self.test_mode = False
        self.environment_mode = "space"
        self.earth_day_duration = 6 * 60
        self.weather_cloud_cover = 0.18
        self.weather_humidity = 0.45

    def set_environment_mode(self, mode: str) -> None:
        if mode not in ENVIRONMENT_MODES:
            raise ValueError(f"Unsupported environment mode: {mode}")
        if mode == self.environment_mode:
            return
        self.environment_mode = mode
        self.manual_sunlight = None
        self.start_time = (
            time.time() - self.earth_day_duration * 0.5
            if mode == "earth"
            else time.time()
        )
        self.battery_charge = 72.0 if mode == "space" else 58.0
        self.total_energy_wh = 0.0
        self.surface_condition = [100.0 for _ in range(self.array_count)]
        self.degradation = [0.0 for _ in range(self.array_count)]
        self.time_since_event = [900.0 for _ in range(self.array_count)]
        self.weather_cloud_cover = 0.18
        self.weather_humidity = 0.45

    def is_sunlight(self) -> bool:
        if self.manual_sunlight is not None:
            return self.manual_sunlight
        if self.environment_mode == "earth":
            day_position = ((time.time() - self.start_time) % self.earth_day_duration) / self.earth_day_duration
            return 0.24 <= day_position <= 0.78
        elapsed = (time.time() - self.start_time) % (self.sunlight_duration + self.shadow_duration)
        return elapsed < self.sunlight_duration

    def time_until_orbit_change(self) -> int:
        if self.manual_sunlight is not None:
            return 0
        if self.environment_mode == "earth":
            elapsed = (time.time() - self.start_time) % self.earth_day_duration
            sunrise = self.earth_day_duration * 0.24
            sunset = self.earth_day_duration * 0.78
            if elapsed < sunrise:
                return int(sunrise - elapsed)
            if elapsed < sunset:
                return int(sunset - elapsed)
            return int(self.earth_day_duration - elapsed + sunrise)
        period = self.sunlight_duration + self.shadow_duration
        elapsed = (time.time() - self.start_time) % period
        if elapsed < self.sunlight_duration:
            return int(self.sunlight_duration - elapsed)
        return int(period - elapsed)

    def force_orbit(self, mode: str) -> None:
        if mode == "sunlight":
            self.manual_sunlight = True
        elif mode == "shadow":
            self.manual_sunlight = False
        else:
            self.manual_sunlight = None

    def inject_event(self, array_id: int, label: str) -> None:
        self.forced_events[array_id] = label

    def _choose_event(self, array_id: int) -> str:
        forced = self.forced_events.pop(array_id, None)
        if forced:
            return forced

        if self.environment_mode == "earth":
            probabilities = {
                "PARTIAL_SHADOW": 0.055 if self.test_mode else 0.009,
                "CLOUD_COVER": 0.065 if self.test_mode else 0.012,
                "SOILING": 0.035 if self.test_mode else 0.004,
                "BENCHMARK_UNDERPERFORMANCE": 0.045 if self.test_mode else 0.006,
                "STRING_FAULT": 0.018 if self.test_mode else 0.0018,
                "INVERTER_CLIPPING": 0.025 if self.test_mode else 0.0035,
                "OVERHEAT": 0.018 if self.test_mode else 0.0018,
            }
        else:
            probabilities = SPACE_TEST_EVENT_PROBABILITIES if self.test_mode else SPACE_EVENT_PROBABILITIES
        draw = random.random()
        cursor = 0.0
        for label, probability in probabilities.items():
            cursor += probability
            if draw < cursor:
                return label
        return "NORMAL"

    def next_readings(self, dt_seconds: float = 1.0) -> tuple[list[SolarPanelReading], dict]:
        self._update_earth_weather(dt_seconds)
        sunlight = self.is_sunlight()
        readings = [self._reading_for_array(i + 1, sunlight, dt_seconds) for i in range(self.array_count)]

        total_power = sum(row.power for row in readings)
        average_efficiency = sum(row.efficiency for row in readings) / len(readings)
        load_target = (
            STARLAB_BASE_LOAD_W + STARLAB_PAYLOAD_POWER_W
            if self.environment_mode == "space"
            else 18_500.0
        )
        excess_power = max(0.0, total_power - load_target)
        fixed_charge_power = total_power * ISS_FIXED_BATTERY_FRACTION
        dynamic_charge_power = 0.0
        payload_power = min(STARLAB_PAYLOAD_POWER_W, max(0.0, total_power - STARLAB_BASE_LOAD_W))
        science_headroom = max(0.0, total_power - STARLAB_BASE_LOAD_W - dynamic_charge_power)
        battery_priority_fraction = 0.0

        system_loads = {}
        if self.environment_mode == "earth":
            if sunlight and excess_power > 1200 and average_efficiency >= 18:
                dynamic_charge_power = min(excess_power * 0.75, 7_500.0)
                battery_priority_fraction = dynamic_charge_power / max(total_power, 1.0)
                self.battery_charge = min(100.0, self.battery_charge + 0.025 * dt_seconds)
                power_mode = "BATTERY CHARGE"
            elif sunlight:
                dynamic_charge_power = max(0.0, excess_power * 0.35)
                battery_priority_fraction = dynamic_charge_power / max(total_power, 1.0)
                self.battery_charge = min(100.0, self.battery_charge + 0.006 * dt_seconds)
                power_mode = "SITE LOAD"
            else:
                self.battery_charge = max(12.0, self.battery_charge - 0.018 * dt_seconds)
                power_mode = "NIGHT RESERVE"
        elif sunlight:
            eclipse_countdown = self.time_until_orbit_change()
            charge_multiplier = 1.0
            if eclipse_countdown < 10 * 60 or self.battery_charge < 55:
                charge_multiplier = 1.25
                power_mode = "ECLIPSE PREP"
            elif excess_power > 7_000 and average_efficiency >= 28:
                power_mode = "BATTERY PRIORITY"
            else:
                power_mode = "SCIENCE PRIORITY"
            dynamic_charge_power = min(excess_power * 0.82 * charge_multiplier, max(excess_power, 0.0))
            battery_priority_fraction = dynamic_charge_power / max(total_power, 1.0)
            self.battery_charge = min(100.0, self.battery_charge + (dynamic_charge_power / 450_000.0) * dt_seconds)
        else:
            dynamic_charge_power = 0.0
            battery_priority_fraction = 0.0
            payload_power = min(STARLAB_PAYLOAD_POWER_W, max(0.0, self.battery_charge - 18.0) * 420.0)
            self.battery_charge = max(18.0, self.battery_charge - 0.025 * dt_seconds)
            power_mode = "SHADOW RESERVE"

        science_headroom = max(0.0, total_power - STARLAB_BASE_LOAD_W - dynamic_charge_power)
        if self.environment_mode == "earth":
            payload_power = 0.0
            science_headroom = max(0.0, total_power - 18_500.0 - dynamic_charge_power)
            site_load = min(18_500.0, max(0.0, total_power - dynamic_charge_power))
            system_loads = {
                "site_load": site_load,
                "inverter": site_load * 0.08,
                "battery_controller": max(dynamic_charge_power, 0.0),
            }
            power_margin = max(0.0, total_power - dynamic_charge_power - site_load)
        else:
            available_after_charge = max(0.0, total_power - dynamic_charge_power)
            critical_power = min(STARLAB_BASE_LOAD_W, available_after_charge)
            non_payload_base = STARLAB_BASE_LOAD_W
            critical_scale = critical_power / non_payload_base if non_payload_base else 0.0
            payload_power = min(payload_power, max(0.0, available_after_charge - critical_power))
            payload_scale = payload_power / STARLAB_PAYLOAD_POWER_W if STARLAB_PAYLOAD_POWER_W else 0.0
            system_loads = {
                "systems_deck": 12_400.0 * critical_scale,
                "payload_lab": 8_600.0 * payload_scale,
                "crew_hab": 4_200.0 * critical_scale,
                "utility_bay": 4_600.0 * critical_scale,
                "external_payloads": 3_400.0 * payload_scale,
                "thermal": 5_300.0 * critical_scale,
                "propulsion": 2_100.0 * critical_scale,
                "docking_robotics": 1_900.0 * critical_scale,
                "stowage_upper": 320.0 * critical_scale,
                "stowage_mid": 360.0 * critical_scale,
                "stowage_low": 420.0 * critical_scale,
                "airlock": 900.0 * payload_scale,
            }
            used_loads = sum(system_loads.values())
            system_loads["unallocated_margin"] = max(0.0, available_after_charge - used_loads)
            science_headroom = max(0.0, payload_power + system_loads["unallocated_margin"])
            power_margin = system_loads["unallocated_margin"]

        battery_signed_power = dynamic_charge_power if sunlight else -load_target * 0.58
        nominal_pack_voltage = 760.0 if self.environment_mode == "space" else 420.0
        battery_pack_voltage = nominal_pack_voltage * (0.88 + 0.22 * self.battery_charge / 100.0)
        battery_pack_current = battery_signed_power / max(battery_pack_voltage, 1.0)
        battery_heat_w = abs(battery_signed_power) * 0.018 + abs(battery_pack_current) * 0.9
        battery_pack_temp = 22.0 + battery_heat_w / 950.0 + (self.battery_charge - 55.0) * 0.025
        if self.environment_mode == "space" and not sunlight:
            battery_pack_temp -= 7.5

        self.total_energy_wh += total_power * dt_seconds / 3600.0
        metrics = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "total_power": total_power,
            "average_efficiency": average_efficiency,
            "total_energy_wh": self.total_energy_wh,
            "battery_charge": self.battery_charge,
            "power_mode": power_mode,
            "excess_power": excess_power,
            "fixed_charge_power": fixed_charge_power,
            "dynamic_charge_power": dynamic_charge_power,
            "battery_priority_fraction": battery_priority_fraction,
            "payload_power": payload_power,
            "science_headroom": science_headroom,
            "power_margin": power_margin,
            "system_loads": system_loads,
            "battery_signed_power": battery_signed_power,
            "battery_pack_voltage": battery_pack_voltage,
            "battery_pack_current": battery_pack_current,
            "battery_pack_temp": battery_pack_temp,
            "starlab_rated_power": STARLAB_RATED_POWER_W,
            "payload_target": STARLAB_PAYLOAD_POWER_W,
            "base_load": STARLAB_BASE_LOAD_W if self.environment_mode == "space" else 18_500.0,
            "sunlight": sunlight,
            "time_to_orbit_change": self.time_until_orbit_change(),
            "environment_mode": self.environment_mode,
            "cloud_cover": self.weather_cloud_cover if self.environment_mode == "earth" else 0.0,
            "humidity": self.weather_humidity if self.environment_mode == "earth" else 0.0,
        }
        return readings, metrics

    def _reading_for_array(self, array_id: int, sunlight: bool, dt_seconds: float) -> SolarPanelReading:
        if self.environment_mode == "earth":
            return self._earth_reading_for_array(array_id, sunlight, dt_seconds)
        return self._space_reading_for_array(array_id, sunlight, dt_seconds)

    def _space_reading_for_array(self, array_id: int, sunlight: bool, dt_seconds: float) -> SolarPanelReading:
        i = array_id - 1
        event = self._choose_event(array_id)
        self.time_since_event[i] += dt_seconds
        self.degradation[i] = min(18.0, self.degradation[i] + random.uniform(0.00001, 0.00008) * dt_seconds)
        self.surface_condition[i] = max(72.0, self.surface_condition[i] - random.uniform(0.0004, 0.0014) * dt_seconds)

        panel_angle = random.uniform(-18, 18)
        exposure_factor = max(0.18, math.cos(math.radians(panel_angle))) if sunlight else 0.02
        irradiance = random.uniform(1285, 1375) * exposure_factor if sunlight else random.uniform(0, 8)
        voltage = random.uniform(154, 168) if sunlight else random.uniform(58, 92)
        panel_area = 37.0
        nominal_efficiency = random.uniform(0.295, 0.325)
        baseline_power = irradiance * panel_area * nominal_efficiency
        current = max(0.05, baseline_power / max(voltage, 1.0) * random.uniform(0.96, 1.04))
        temperature = random.uniform(38, 72) if sunlight else random.uniform(-82, -28)
        radiation_level = random.uniform(38, 64)
        vibration_g = random.uniform(0.004, 0.035)

        if event != "NORMAL":
            self.time_since_event[i] = 0.0

        if event == "SMALL_IMPACT":
            voltage *= random.uniform(0.86, 0.93)
            current *= random.uniform(0.82, 0.9)
            vibration_g = random.uniform(0.18, 0.85)
            self.surface_condition[i] = max(68.0, self.surface_condition[i] - random.uniform(1.5, 4.5))
        elif event == "LARGE_IMPACT":
            voltage *= random.uniform(0.58, 0.72)
            current *= random.uniform(0.48, 0.68)
            temperature += random.uniform(20, 65)
            vibration_g = random.uniform(1.4, 6.2)
            self.surface_condition[i] = max(45.0, self.surface_condition[i] - random.uniform(10, 24))
        elif event == "PARTIAL_SHADOW":
            irradiance *= random.uniform(0.35, 0.62)
            current *= random.uniform(0.45, 0.7)
        elif event == "DUST":
            self.surface_condition[i] = max(62.0, self.surface_condition[i] - random.uniform(2, 8))
            irradiance *= random.uniform(0.72, 0.88)
        elif event == "OVERHEAT":
            temperature = random.uniform(82, 135)
            voltage *= random.uniform(0.78, 0.9)
        elif event == "RADIATION":
            radiation_level = random.uniform(220, 540)
            voltage *= random.uniform(0.88, 0.96)

        power = voltage * current
        if irradiance > 0:
            efficiency = max(0.0, min(38.0, (power / (irradiance * panel_area)) * 100.0))
        else:
            efficiency = 0.0
        efficiency -= (100.0 - self.surface_condition[i]) * 0.03 + self.degradation[i] * 0.04
        efficiency += math.sin(time.time() / 12 + array_id) * 0.35
        efficiency = max(0.0, min(38.0, efficiency))

        return SolarPanelReading(
            array_id=array_id,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            voltage=voltage,
            current=current,
            power=power,
            temperature=temperature,
            irradiance=irradiance,
            panel_angle=panel_angle,
            efficiency=efficiency,
            radiation_level=radiation_level,
            surface_condition=self.surface_condition[i],
            storage_voltage=random.uniform(116, 124),
            storage_current=current * random.uniform(0.84, 0.96),
            degradation=self.degradation[i],
            time_since_event=self.time_since_event[i],
            sunlight=sunlight,
            environment_mode=self.environment_mode,
            vibration_g=vibration_g,
            exposure_factor=exposure_factor,
            benchmark_power=baseline_power,
            actual_label=event,
        )

    def _update_earth_weather(self, dt_seconds: float) -> None:
        if self.environment_mode != "earth":
            return
        self.weather_cloud_cover += random.uniform(-0.012, 0.012) * max(1.0, dt_seconds / 2)
        self.weather_humidity += random.uniform(-0.01, 0.01) * max(1.0, dt_seconds / 2)
        if random.random() < 0.015:
            self.weather_cloud_cover += random.uniform(0.12, 0.32)
        if random.random() < 0.01:
            self.weather_cloud_cover -= random.uniform(0.08, 0.2)
        self.weather_cloud_cover = max(0.02, min(0.92, self.weather_cloud_cover))
        self.weather_humidity = max(0.18, min(0.95, self.weather_humidity))

    def _earth_sun_factor(self) -> float:
        if self.manual_sunlight is True:
            return 0.9
        if self.manual_sunlight is False:
            return 0.0
        day_position = ((time.time() - self.start_time) % self.earth_day_duration) / self.earth_day_duration
        daylight_start = 0.24
        daylight_end = 0.78
        if day_position < daylight_start or day_position > daylight_end:
            return 0.0
        daylight_progress = (day_position - daylight_start) / (daylight_end - daylight_start)
        return max(0.02, math.sin(math.pi * daylight_progress))

    def _earth_reading_for_array(self, array_id: int, sunlight: bool, dt_seconds: float) -> SolarPanelReading:
        i = array_id - 1
        event = self._choose_event(array_id)
        self.time_since_event[i] += dt_seconds
        self.degradation[i] = min(28.0, self.degradation[i] + random.uniform(0.00004, 0.00018) * dt_seconds)
        self.surface_condition[i] = max(58.0, self.surface_condition[i] - random.uniform(0.001, 0.004) * dt_seconds)

        sun_factor = self._earth_sun_factor()
        cloud_loss = 1.0 - self.weather_cloud_cover * random.uniform(0.45, 0.85)
        exposure_factor = sun_factor * cloud_loss
        irradiance = max(0.0, random.uniform(860, 1035) * sun_factor * cloud_loss)
        if not sunlight:
            irradiance = random.uniform(0, 4)

        ambient = random.uniform(7, 34)
        temperature = ambient + irradiance * random.uniform(0.025, 0.045)
        voltage = random.uniform(365, 485) if sunlight else random.uniform(80, 160)
        current = max(0.05, (irradiance / 1000.0) * random.uniform(11.0, 15.0))
        benchmark_current = max(0.05, (irradiance / 1000.0) * 13.2)
        benchmark_power = voltage * benchmark_current
        panel_angle = random.uniform(18, 38) + math.sin(time.time() / 35 + array_id) * 6
        radiation_level = max(0.0, min(12.0, (irradiance / 1000.0) * random.uniform(6.0, 10.5)))
        vibration_g = random.uniform(0.002, 0.018)

        if event != "NORMAL":
            self.time_since_event[i] = 0.0
        if event not in {"NORMAL", "CLOUD_COVER"}:
            self.weather_cloud_cover = min(self.weather_cloud_cover, random.uniform(0.08, 0.24))

        if event == "PARTIAL_SHADOW":
            irradiance *= random.uniform(0.18, 0.48)
            current *= random.uniform(0.22, 0.55)
        elif event == "CLOUD_COVER":
            self.weather_cloud_cover = min(0.94, self.weather_cloud_cover + random.uniform(0.18, 0.38))
            irradiance *= random.uniform(0.38, 0.68)
            current *= random.uniform(0.42, 0.7)
            benchmark_power = voltage * max(0.05, (irradiance / 1000.0) * 13.2)
        elif event == "SOILING":
            self.surface_condition[i] = max(48.0, self.surface_condition[i] - random.uniform(5, 14))
            current *= random.uniform(0.68, 0.84)
        elif event == "BENCHMARK_UNDERPERFORMANCE":
            current *= random.uniform(0.45, 0.62)
            voltage *= random.uniform(0.92, 0.98)
        elif event == "STRING_FAULT":
            current *= random.uniform(0.18, 0.42)
            voltage *= random.uniform(0.74, 0.88)
        elif event == "INVERTER_CLIPPING":
            clip_power = random.uniform(3600, 4700)
            current = min(current, clip_power / max(voltage, 1.0))
        elif event == "OVERHEAT":
            temperature = random.uniform(72, 94)
            voltage *= random.uniform(0.84, 0.92)

        power = voltage * current
        if irradiance > 10 and benchmark_power > 1:
            benchmark_efficiency = random.uniform(18.2, 21.8)
            performance_ratio = max(0.0, min(1.15, power / benchmark_power))
            efficiency = benchmark_efficiency * performance_ratio
        else:
            efficiency = 0.0
        efficiency -= (100.0 - self.surface_condition[i]) * 0.025 + self.degradation[i] * 0.025
        efficiency = max(0.0, min(25.0, efficiency))

        return SolarPanelReading(
            array_id=array_id,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            voltage=voltage,
            current=current,
            power=power,
            temperature=temperature,
            irradiance=irradiance,
            panel_angle=panel_angle,
            efficiency=efficiency,
            radiation_level=radiation_level,
            surface_condition=self.surface_condition[i],
            storage_voltage=random.uniform(46, 54),
            storage_current=current * random.uniform(0.72, 0.92),
            degradation=self.degradation[i],
            time_since_event=self.time_since_event[i],
            sunlight=sunlight,
            environment_mode=self.environment_mode,
            vibration_g=vibration_g,
            exposure_factor=exposure_factor,
            cloud_cover=self.weather_cloud_cover,
            benchmark_power=benchmark_power,
            actual_label=event,
        )


def synthesize_dataset(
    samples: int = 24000, seed: int = 7
) -> tuple[list[list[float]], list[int]]:
    """Generate labeled panel readings for training in both environments."""
    random.seed(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    simulator = SolarArraySimulator()
    class_to_index = {name: index for index, name in enumerate(ANOMALY_CLASSES)}

    for index in range(samples):
        # Alternate environments to keep both represented in the training set.
        is_earth = index % 2 == 1
        mode = "earth" if is_earth else "space"
        classes = EARTH_ANOMALY_CLASSES if is_earth else SPACE_ANOMALY_CLASSES
        label = classes[(index // 2) % len(classes)]
        simulator.set_environment_mode(mode)
        simulator.inject_event(1, label)
        darkness_chance = 0.12 if is_earth else 0.18
        simulator.manual_sunlight = random.random() > darkness_chance
        reading = simulator._reading_for_array(1, simulator.is_sunlight(), 1.0)
        rows.append(reading.features())
        labels.append(class_to_index[reading.actual_label])

    labeled_rows = list(zip(rows, labels))
    random.shuffle(labeled_rows)
    shuffled_rows, shuffled_labels = zip(*labeled_rows)
    return list(shuffled_rows), list(shuffled_labels)
