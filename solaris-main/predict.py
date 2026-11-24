from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from database import (
    acknowledge_all,
    database_counts,
    init_database,
    insert_incident,
    insert_metric,
    insert_readings,
    recent_incidents,
    recent_readings,
)
from ml_model import StarlabAnomalyModel, cuda_summary, train_and_save
from simulator import IMPACT_CLASSES, STARLAB_PAYLOAD_POWER_W, STARLAB_RATED_POWER_W, SolarArraySimulator


COLORS = {
    "bg": "#080b0f",
    "panel": "#101720",
    "panel_alt": "#151f2b",
    "line": "#253448",
    "text": "#eaf2ff",
    "muted": "#8fa3b8",
    "green": "#31d17c",
    "yellow": "#f2cc4d",
    "red": "#ff5666",
    "cyan": "#25c7da",
    "blue": "#4f8cff",
}

ARRAY_COLORS = ["#25c7da", "#a77cff", "#31d17c", "#f2cc4d"]
SEVERITY = {
    "SMALL_IMPACT": ("HIGH", "Micrometeoroid or debris impact detected"),
    "LARGE_IMPACT": ("CRITICAL", "Major panel impact detected. Inspect array immediately"),
    "PARTIAL_SHADOW": ("MEDIUM", "Unexpected panel shadowing is reducing power"),
    "DUST": ("LOW", "Surface contamination is reducing solar conversion"),
    "OVERHEAT": ("HIGH", "Thermal excursion detected on panel electronics"),
    "RADIATION": ("HIGH", "Elevated radiation is interfering with panel output"),
    "CLOUD_COVER": ("MEDIUM", "Cloud cover is reducing solar production below forecast"),
    "SOILING": ("LOW", "Soiling buildup is reducing terrestrial panel output"),
    "BENCHMARK_UNDERPERFORMANCE": ("MEDIUM", "Output is below irradiance and temperature benchmark"),
    "STRING_FAULT": ("HIGH", "PV string current is far below expected output"),
    "INVERTER_CLIPPING": ("LOW", "Inverter clipping is limiting peak output"),
}


class StarlabDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Starlab Solar Energy Monitor")
        self.root.geometry("1460x900")
        self.root.minsize(1240, 760)
        self.root.configure(bg=COLORS["bg"])

        init_database()
        self.simulator = SolarArraySimulator()
        self.model = StarlabAnomalyModel()
        self.model_ready = False
        self.model_status = "Loading model..."
        self.ui_queue: queue.Queue[str] = queue.Queue()
        self.paused = False
        self.sim_speed = tk.DoubleVar(value=1.0)
        self.power_history: list[float] = []
        self.efficiency_history: list[float] = []
        self.last_incident: dict[tuple[int, str], float] = {}
        self.latest_readings = []
        self.latest_predictions: dict[int, tuple[str, float]] = {}
        self.array_labels: dict[int, dict[str, tk.Label]] = {}
        self.environment_var = tk.StringVar(value="Space Station")
        self.routing_labels: dict[str, tk.Label] = {}
        self.mission_labels: dict[str, tk.Label] = {}
        self.mesh_tree: ttk.Treeview | None = None
        self.last_metrics: dict = {}
        self.battery_window: tk.Toplevel | None = None
        self.battery_canvas: tk.Canvas | None = None
        self.battery_detail_label: tk.Label | None = None
        self.battery_metric_labels: dict[str, tk.Label] = {}
        self.battery_selected = "overview"
        self.battery_regions: dict[str, tuple[int, int, int, int]] = {}
        self.main_tabs: ttk.Notebook | None = None
        self.panel_tab: tk.Frame | None = None
        self.battery_tab: tk.Frame | None = None
        self.system_flow_tab: tk.Frame | None = None
        self.benchmark_tab: tk.Frame | None = None
        self.system_flow_canvas: tk.Canvas | None = None
        self.system_flow_detail_label: tk.Label | None = None
        self.system_flow_metric_labels: dict[str, tk.Label] = {}
        self.system_flow_selected = "overview"
        self.system_flow_regions: dict[str, tuple[int, int, int, int]] = {}
        self.benchmark_labels: dict[str, tk.Label] = {}
        self.benchmark_extra_payload_wh = 0.0
        self.benchmark_avoided_eva_events: set[tuple[int, str]] = set()

        self._sound = self._load_sound()
        self._configure_style()
        self._build_ui()
        self._load_model_async()
        self._tick()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), background=COLORS["panel"], foreground=COLORS["text"])
        style.map("TNotebook.Tab", background=[("selected", COLORS["panel_alt"])])
        style.configure("Treeview", background=COLORS["panel"], fieldbackground=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Treeview.Heading", background=COLORS["panel_alt"], foreground=COLORS["text"])

    def _build_ui(self) -> None:
        self.header = tk.Frame(self.root, bg=COLORS["bg"], padx=18, pady=14)
        self.header.pack(fill="x")

        title_block = tk.Frame(self.header, bg=COLORS["bg"])
        title_block.pack(side="left")
        tk.Label(
            title_block,
            text="STANDBY SOLAR SENTINEL",
            fg=COLORS["muted"],
            bg=COLORS["bg"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="Starlab Solar Energy Monitor",
            fg=COLORS["text"],
            bg=COLORS["bg"],
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w")

        self.status_label = self._header_value("System", "Initializing")
        self.mode_label = self._header_value("Power Mode", "--")
        self.battery_label = self._header_value("Battery", "--")
        self.environment_label = self._header_value("Environment", "Space")
        self.device_label = self._header_value("ML Device", cuda_summary())

        controls = tk.Frame(self.root, bg=COLORS["panel"], padx=14, pady=10)
        controls.pack(fill="x", padx=18)
        self.pause_button = self._button(controls, "Pause", self._toggle_pause)
        self.pause_button.pack(side="left", padx=(0, 8))
        self._button(controls, "Acknowledge Alerts", self._acknowledge_alerts).pack(side="left", padx=8)
        self._button(controls, "Battery Diagram", self._select_battery_tab).pack(side="left", padx=8)
        self._button(controls, "Station Flow", self._select_system_flow_tab).pack(side="left", padx=8)
        self._button(controls, "Benchmarks", self._select_benchmark_tab).pack(side="left", padx=8)
        self._button(controls, "Retrain CUDA Model", self._retrain_async).pack(side="left", padx=8)

        tk.Label(controls, text="Mode", fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 10)).pack(
            side="left", padx=(22, 6)
        )
        environment = ttk.Combobox(
            controls,
            textvariable=self.environment_var,
            values=("Space Station", "Earth Solar Farm"),
            width=17,
            state="readonly",
        )
        environment.pack(side="left")
        environment.bind("<<ComboboxSelected>>", lambda _event: self._change_environment())

        tk.Label(controls, text="Light", fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 10)).pack(
            side="left", padx=(16, 6)
        )
        self.orbit_var = tk.StringVar(value="auto")
        orbit = ttk.Combobox(
            controls,
            textvariable=self.orbit_var,
            values=("auto", "sunlight", "shadow"),
            width=9,
            state="readonly",
        )
        orbit.pack(side="left")
        orbit.bind("<<ComboboxSelected>>", lambda _event: self.simulator.force_orbit(self.orbit_var.get()))

        self.test_var = tk.BooleanVar(value=False)
        test = tk.Checkbutton(
            controls,
            text="Test anomaly mode",
            variable=self.test_var,
            command=self._toggle_test_mode,
            fg=COLORS["text"],
            bg=COLORS["panel"],
            selectcolor=COLORS["panel_alt"],
            activebackground=COLORS["panel"],
            activeforeground=COLORS["text"],
        )
        test.pack(side="left", padx=12)

        tk.Label(controls, text="Sim speed", fg=COLORS["muted"], bg=COLORS["panel"]).pack(side="left", padx=(10, 4))
        tk.Scale(
            controls,
            variable=self.sim_speed,
            from_=1,
            to=20,
            resolution=1,
            orient="horizontal",
            length=115,
            fg=COLORS["text"],
            bg=COLORS["panel"],
            troughcolor=COLORS["line"],
            highlightthickness=0,
        ).pack(side="left")

        self.main_tabs = ttk.Notebook(self.root)
        self.main_tabs.pack(fill="both", expand=True, padx=18, pady=14)
        self.panel_tab = tk.Frame(self.main_tabs, bg=COLORS["bg"])
        self.battery_tab = tk.Frame(self.main_tabs, bg=COLORS["bg"])
        self.system_flow_tab = tk.Frame(self.main_tabs, bg=COLORS["bg"])
        self.benchmark_tab = tk.Frame(self.main_tabs, bg=COLORS["bg"])
        self.main_tabs.add(self.panel_tab, text="Panel Monitoring")
        self.main_tabs.add(self.battery_tab, text="Battery")
        self.main_tabs.add(self.system_flow_tab, text="System Energy Flow")
        self.main_tabs.add(self.benchmark_tab, text="Benchmarks")

        body = tk.Frame(self.panel_tab, bg=COLORS["bg"])
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=23)
        body.columnconfigure(1, weight=36)
        body.columnconfigure(2, weight=22)
        body.rowconfigure(0, weight=1)

        self.array_frame = tk.Frame(body, bg=COLORS["bg"])
        self.array_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        for row in range(2):
            self.array_frame.rowconfigure(row, weight=1)
            self.array_frame.columnconfigure(row, weight=1)
        for index in range(1, 5):
            self._build_array_card(index)

        center = tk.Frame(body, bg=COLORS["bg"])
        center.grid(row=0, column=1, sticky="nsew", padx=12)
        center.rowconfigure(0, weight=3)
        center.rowconfigure(1, weight=2)
        center.columnconfigure(0, weight=1)
        self.network_canvas = tk.Canvas(center, bg=COLORS["panel"], highlightthickness=1, highlightbackground=COLORS["line"])
        self.network_canvas.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        self.chart_canvas = tk.Canvas(center, bg=COLORS["panel"], highlightthickness=1, highlightbackground=COLORS["line"])
        self.chart_canvas.grid(row=1, column=0, sticky="nsew")

        self.side_tabs = ttk.Notebook(body)
        self.side_tabs.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        self._build_event_tab()
        self._build_routing_tab()
        self._build_mesh_tab()
        self._build_database_tab()
        self._build_model_tab()
        self._build_battery_tab()
        self._build_system_flow_tab()
        self._build_benchmark_tab()

    def _header_value(self, label: str, value: str) -> tk.Label:
        box = tk.Frame(self.header, bg=COLORS["panel"], padx=10, pady=7)
        box.pack(side="right", padx=4)
        tk.Label(box, text=label, fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 9)).pack(anchor="w")
        value_label = tk.Label(
            box,
            text=value,
            fg=COLORS["text"],
            bg=COLORS["panel"],
            font=("Segoe UI", 10, "bold"),
            wraplength=165,
            justify="left",
        )
        value_label.pack(anchor="w")
        return value_label

    def _button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            fg=COLORS["text"],
            bg=COLORS["panel_alt"],
            activebackground=COLORS["line"],
            activeforeground=COLORS["text"],
            relief="flat",
            padx=9,
            pady=6,
            font=("Segoe UI", 8, "bold"),
        )

    def _build_array_card(self, array_id: int) -> None:
        card = tk.Frame(
            self.array_frame,
            bg=COLORS["panel"],
            highlightbackground=COLORS["line"],
            highlightthickness=1,
            padx=12,
            pady=10,
        )
        row, column = divmod(array_id - 1, 2)
        card.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
        card.columnconfigure(1, weight=1)

        tk.Label(
            card,
            text=f"Array {array_id}",
            fg=ARRAY_COLORS[array_id - 1],
            bg=COLORS["panel"],
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        labels: dict[str, tk.Label] = {}
        metrics = ["Power", "Efficiency", "Voltage", "Current", "Temp", "Exposure", "Vibration", "Prediction"]
        for row_index, metric in enumerate(metrics, start=1):
            tk.Label(card, text=metric, fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 9)).grid(
                row=row_index, column=0, sticky="w", pady=2
            )
            value = tk.Label(
                card,
                text="--",
                fg=COLORS["text"],
                bg=COLORS["panel"],
                font=("Segoe UI", 9, "bold"),
                wraplength=118,
                justify="right",
            )
            value.grid(row=row_index, column=1, sticky="e", pady=2)
            labels[metric] = value

        actions = tk.Frame(card, bg=COLORS["panel"])
        actions.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.array_labels[array_id] = labels
        labels["Actions"] = actions
        self._populate_array_actions(array_id)

    def _populate_array_actions(self, array_id: int) -> None:
        labels = self.array_labels.get(array_id)
        if not labels:
            return
        actions = labels["Actions"]
        for child in actions.winfo_children():
            child.destroy()
        if self.simulator.environment_mode == "earth":
            buttons = (
                ("PARTIAL_SHADOW", "Shade"),
                ("CLOUD_COVER", "Cloud"),
                ("BENCHMARK_UNDERPERFORMANCE", "Bench"),
                ("STRING_FAULT", "String"),
            )
        else:
            buttons = (
                ("SMALL_IMPACT", "Small"),
                ("LARGE_IMPACT", "Major"),
                ("PARTIAL_SHADOW", "Shade"),
            )
        for label, text in buttons:
            self._button(
                actions,
                text,
                lambda event_label=label, aid=array_id: self.simulator.inject_event(aid, event_label),
            ).pack(side="left", expand=True, fill="x", padx=2)

    def _build_event_tab(self) -> None:
        tab = tk.Frame(self.side_tabs, bg=COLORS["panel"], padx=10, pady=10)
        self.side_tabs.add(tab, text="Events")
        self.alert_label = tk.Label(
            tab,
            text="No active alerts",
            fg=COLORS["green"],
            bg=COLORS["panel"],
            font=("Segoe UI", 13, "bold"),
            wraplength=300,
            justify="left",
        )
        self.alert_label.pack(fill="x", pady=(0, 10))

        self.incident_tree = ttk.Treeview(tab, columns=("time", "array", "type", "severity"), show="headings", height=18)
        for column, width in (("time", 82), ("array", 48), ("type", 116), ("severity", 72)):
            self.incident_tree.heading(column, text=column.title())
            self.incident_tree.column(column, width=width, anchor="w")
        self.incident_tree.pack(fill="both", expand=True)

    def _build_routing_tab(self) -> None:
        tab = tk.Frame(self.side_tabs, bg=COLORS["panel"], padx=10, pady=10)
        self.side_tabs.add(tab, text="Routing")
        facts = (
            ("Rated Generation", f"{STARLAB_RATED_POWER_W / 1000:.0f} kW"),
            ("Payload Target", f"{STARLAB_PAYLOAD_POWER_W / 1000:.0f} kW"),
            ("Local Loop", "On-board DB + ML"),
            ("Fallback", "standard routing baseline"),
        )
        for label, value in facts:
            row = tk.Frame(tab, bg=COLORS["panel"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 9)).pack(anchor="w")
            tk.Label(row, text=value, fg=COLORS["text"], bg=COLORS["panel"], font=("Segoe UI", 12, "bold")).pack(anchor="w")

        tk.Frame(tab, height=1, bg=COLORS["line"]).pack(fill="x", pady=10)
        for key, label in (
            ("dynamic_charge_power", "Dynamic Battery Charge"),
            ("fixed_charge_power", "ISS Fixed 60% Charge"),
            ("payload_power", "Payload Power"),
            ("science_headroom", "Science Headroom"),
            ("battery_priority_fraction", "Battery Routing Share"),
            ("local_latency_ms", "Local ML Latency"),
        ):
            self.routing_labels[key] = self._metric_row(tab, label)

    def _build_mesh_tab(self) -> None:
        tab = tk.Frame(self.side_tabs, bg=COLORS["panel"], padx=10, pady=10)
        self.side_tabs.add(tab, text="Mesh")
        tk.Label(
            tab,
            text="Sensor mesh health",
            fg=COLORS["text"],
            bg=COLORS["panel"],
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        columns = ("node", "kw", "ratio", "vib", "exposure")
        self.mesh_tree = ttk.Treeview(tab, columns=columns, show="headings", height=10)
        for column, width in (("node", 52), ("kw", 62), ("ratio", 58), ("vib", 54), ("exposure", 68)):
            self.mesh_tree.heading(column, text=column.title())
            self.mesh_tree.column(column, width=width, anchor="w")
        self.mesh_tree.pack(fill="x", pady=(0, 12))
        self.mission_labels["mesh_summary"] = tk.Label(
            tab,
            text="Waiting for telemetry...",
            fg=COLORS["muted"],
            bg=COLORS["panel"],
            justify="left",
            wraplength=305,
            font=("Segoe UI", 10),
        )
        self.mission_labels["mesh_summary"].pack(fill="x", anchor="w")

    def _metric_row(self, parent: tk.Widget, label: str) -> tk.Label:
        row = tk.Frame(parent, bg=COLORS["panel"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, fg=COLORS["muted"], bg=COLORS["panel"], font=("Segoe UI", 9)).pack(anchor="w")
        value = tk.Label(row, text="--", fg=COLORS["text"], bg=COLORS["panel"], font=("Segoe UI", 11, "bold"))
        value.pack(anchor="w")
        return value

    def _build_database_tab(self) -> None:
        tab = tk.Frame(self.side_tabs, bg=COLORS["panel"], padx=10, pady=10)
        self.side_tabs.add(tab, text="Database")
        self.db_summary = tk.Label(tab, text="", fg=COLORS["text"], bg=COLORS["panel"], justify="left", font=("Consolas", 10))
        self.db_summary.pack(fill="x", anchor="w")
        self.reading_list = tk.Listbox(
            tab,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            highlightthickness=0,
            borderwidth=0,
            font=("Consolas", 9),
        )
        self.reading_list.pack(fill="both", expand=True, pady=(10, 0))

    def _build_model_tab(self) -> None:
        tab = tk.Frame(self.side_tabs, bg=COLORS["panel"], padx=10, pady=10)
        self.side_tabs.add(tab, text="Model")
        self.model_label = tk.Label(
            tab,
            text=self.model_status,
            fg=COLORS["text"],
            bg=COLORS["panel"],
            justify="left",
            wraplength=305,
            font=("Segoe UI", 11),
        )
        self.model_label.pack(fill="x", anchor="w")
        tk.Label(
            tab,
            text=(
                "Local PyTorch classifier\n"
                "Inputs: environment, cloud estimate, benchmark ratio, voltage, current, power, "
                "temperature, irradiance, angle, efficiency, vibration, exposure, surface condition, "
                "storage telemetry, degradation, and event age.\n\n"
                "Space outputs include impacts and radiation. Earth outputs include clouds, shade, "
                "soiling, string faults, clipping, and benchmark underperformance."
            ),
            fg=COLORS["muted"],
            bg=COLORS["panel"],
            justify="left",
            wraplength=305,
            font=("Segoe UI", 10),
        ).pack(fill="x", pady=(16, 0), anchor="w")

    def _build_battery_tab(self) -> None:
        if self.battery_tab is None:
            return
        self.battery_tab.columnconfigure(0, weight=4)
        self.battery_tab.columnconfigure(1, weight=2)
        self.battery_tab.rowconfigure(0, weight=1)

        self.battery_canvas = tk.Canvas(
            self.battery_tab,
            bg=COLORS["panel"],
            highlightthickness=1,
            highlightbackground=COLORS["line"],
        )
        self.battery_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=0)
        self.battery_canvas.bind("<Button-1>", self._on_battery_click)
        self.battery_canvas.bind("<Configure>", lambda _event: self._draw_battery_diagram())

        side = tk.Frame(self.battery_tab, bg=COLORS["panel"], padx=12, pady=12)
        side.grid(row=0, column=1, sticky="nsew")
        tk.Label(
            side,
            text="Silicon Solid-State Battery",
            fg=COLORS["text"],
            bg=COLORS["panel"],
            font=("Segoe UI", 15, "bold"),
            wraplength=330,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            side,
            text="Click cell layers, modules, BMS, charge controller, or station bus.",
            fg=COLORS["muted"],
            bg=COLORS["panel"],
            font=("Segoe UI", 9),
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self.battery_metric_labels = {}
        for key, label in (
            ("soc", "State of Charge"),
            ("pack_voltage", "Pack Voltage"),
            ("pack_current", "Pack Current"),
            ("pack_power", "Charge/Discharge"),
            ("pack_temp", "Pack Temperature"),
            ("cycle_life", "Projected Cycles"),
            ("thermal", "Thermal Mode"),
            ("latency", "Control Latency"),
        ):
            self.battery_metric_labels[key] = self._metric_row(side, label)

        tk.Frame(side, height=1, bg=COLORS["line"]).pack(fill="x", pady=10)
        self.battery_detail_label = tk.Label(
            side,
            text="",
            fg=COLORS["text"],
            bg=COLORS["panel"],
            justify="left",
            wraplength=340,
            font=("Segoe UI", 10),
        )
        self.battery_detail_label.pack(fill="both", expand=True, anchor="w")

    def _build_system_flow_tab(self) -> None:
        if self.system_flow_tab is None:
            return
        self.system_flow_tab.columnconfigure(0, weight=5)
        self.system_flow_tab.columnconfigure(1, weight=2)
        self.system_flow_tab.rowconfigure(0, weight=1)

        self.system_flow_canvas = tk.Canvas(
            self.system_flow_tab,
            bg=COLORS["panel"],
            highlightthickness=1,
            highlightbackground=COLORS["line"],
        )
        self.system_flow_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.system_flow_canvas.bind("<Button-1>", self._on_system_flow_click)
        self.system_flow_canvas.bind("<Configure>", lambda _event: self._draw_system_flow_diagram())

        side = tk.Frame(self.system_flow_tab, bg=COLORS["panel"], padx=12, pady=12)
        side.grid(row=0, column=1, sticky="nsew")
        tk.Label(
            side,
            text="Whole-Station Energy Flow",
            fg=COLORS["text"],
            bg=COLORS["panel"],
            font=("Segoe UI", 15, "bold"),
            wraplength=330,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            side,
            text="Map-style cutaway based on public Starlab layout: service module, utility bay, Systems Deck, Internal Payload Laboratory, Crew Habitation Deck, upper stowage decks, docking ports, Bishop Airlock, external payload platforms, and SKYMAKER arm.",
            fg=COLORS["muted"],
            bg=COLORS["panel"],
            font=("Segoe UI", 9),
            wraplength=330,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self.system_flow_metric_labels = {}
        for key, label in (
            ("generation", "Solar Generation"),
            ("battery", "Battery Charge Path"),
            ("dc_bus", "Regulated DC Bus"),
            ("payload", "Payload Power"),
            ("critical", "Critical Station Loads"),
            ("thermal", "Thermal/Heat Rejection"),
            ("propulsion", "Propulsion/GNC Reserve"),
            ("margin", "Power Margin"),
        ):
            self.system_flow_metric_labels[key] = self._metric_row(side, label)

        tk.Frame(side, height=1, bg=COLORS["line"]).pack(fill="x", pady=10)
        self.system_flow_detail_label = tk.Label(
            side,
            text="Click a station system block.",
            fg=COLORS["text"],
            bg=COLORS["panel"],
            justify="left",
            wraplength=340,
            font=("Segoe UI", 10),
        )
        self.system_flow_detail_label.pack(fill="both", expand=True, anchor="w")

    def _build_benchmark_tab(self) -> None:
        if self.benchmark_tab is None:
            return
        self.benchmark_tab.columnconfigure(0, weight=1)
        self.benchmark_tab.columnconfigure(1, weight=1)
        self.benchmark_tab.rowconfigure(0, weight=1)

        left = tk.Frame(self.benchmark_tab, bg=COLORS["panel"], padx=18, pady=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = tk.Frame(self.benchmark_tab, bg=COLORS["panel"], padx=18, pady=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        tk.Label(left, text="NASA ISS Baseline", fg=COLORS["text"], bg=COLORS["panel"], font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(
            left,
            text=(
                "Presentation-guide benchmark: ISS-style power management monitors power mostly in bulk, "
                "uses a fixed ~60% battery charging rule during sunlit passes, lacks dedicated panel-strike "
                "detection, and often needs EVA/ground loops for diagnosis."
            ),
            fg=COLORS["muted"],
            bg=COLORS["panel"],
            font=("Segoe UI", 10),
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(4, 14))

        for key, label in (
            ("mode", "Benchmark Mode"),
            ("iss_charge", "ISS Fixed 60% Battery Charge"),
            ("our_charge", "Our Dynamic Battery Charge"),
            ("iss_payload", "ISS-Style Payload Power"),
            ("our_payload", "Our Payload Power"),
            ("extra_payload", "Extra Payload Power Now"),
            ("extra_energy", "Cumulative Extra Payload Energy"),
        ):
            self.benchmark_labels[key] = self._metric_row(left, label)

        tk.Label(right, text="Cost / Risk Savings", fg=COLORS["text"], bg=COLORS["panel"], font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(
            right,
            text=(
                "Cost model uses the guide's EVA estimate: $50k-$100k per EVA hour. "
                "For demo accounting, each localized physical panel impact is counted as one avoided "
                "2-hour inspection window because the crew receives location/severity immediately."
            ),
            fg=COLORS["muted"],
            bg=COLORS["panel"],
            font=("Segoe UI", 10),
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(4, 14))

        for key, label in (
            ("avoided_events", "Localized Impact Events"),
            ("eva_hours", "Estimated EVA Hours Avoided"),
            ("eva_low", "Low Savings @ $50k/hr"),
            ("eva_mid", "Mid Savings @ $75k/hr"),
            ("eva_high", "High Savings @ $100k/hr"),
            ("latency", "Local ML Latency"),
            ("summary", "Mission Benefit"),
        ):
            self.benchmark_labels[key] = self._metric_row(right, label)

    def _load_sound(self):
        try:
            import pygame

            pygame.mixer.init()
            sound_path = Path("chime.wav")
            return pygame.mixer.Sound(str(sound_path)) if sound_path.exists() else None
        except Exception:
            return None

    def _open_battery_diagram(self) -> None:
        if self.battery_window is not None and self.battery_window.winfo_exists():
            self.battery_window.lift()
            self._draw_battery_diagram()
            return

        self.battery_window = tk.Toplevel(self.root)
        self.battery_window.title("Next-Gen Silicon Solid-State Battery Diagram")
        self.battery_window.geometry("1180x760")
        self.battery_window.minsize(980, 640)
        self.battery_window.configure(bg=COLORS["bg"])
        self.battery_window.protocol("WM_DELETE_WINDOW", self._close_battery_diagram)

        shell = tk.Frame(self.battery_window, bg=COLORS["bg"], padx=14, pady=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=4)
        shell.columnconfigure(1, weight=2)
        shell.rowconfigure(0, weight=1)

        self.battery_canvas = tk.Canvas(
            shell,
            bg=COLORS["panel"],
            highlightthickness=1,
            highlightbackground=COLORS["line"],
        )
        self.battery_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.battery_canvas.bind("<Button-1>", self._on_battery_click)
        self.battery_canvas.bind("<Configure>", lambda _event: self._draw_battery_diagram())

        side = tk.Frame(shell, bg=COLORS["panel"], padx=12, pady=12)
        side.grid(row=0, column=1, sticky="nsew")
        tk.Label(
            side,
            text="Battery System",
            fg=COLORS["text"],
            bg=COLORS["panel"],
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")
        tk.Label(
            side,
            text="Click any block in the diagram.",
            fg=COLORS["muted"],
            bg=COLORS["panel"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 10))

        for key, label in (
            ("soc", "State of Charge"),
            ("pack_voltage", "Pack Voltage"),
            ("pack_current", "Pack Current"),
            ("pack_power", "Charge/Discharge"),
            ("pack_temp", "Pack Temperature"),
            ("cycle_life", "Projected Cycles"),
            ("thermal", "Thermal Mode"),
            ("latency", "Control Latency"),
        ):
            self.battery_metric_labels[key] = self._metric_row(side, label)

        tk.Frame(side, height=1, bg=COLORS["line"]).pack(fill="x", pady=10)
        self.battery_detail_label = tk.Label(
            side,
            text="",
            fg=COLORS["text"],
            bg=COLORS["panel"],
            justify="left",
            wraplength=340,
            font=("Segoe UI", 10),
        )
        self.battery_detail_label.pack(fill="both", expand=True, anchor="w")

        self._draw_battery_diagram()

    def _close_battery_diagram(self) -> None:
        if self.battery_window is not None:
            self.battery_window.destroy()
        self.battery_window = None
        self.battery_canvas = None
        self.battery_detail_label = None
        self.battery_metric_labels = {}

    def _on_battery_click(self, event) -> None:
        for name, (x1, y1, x2, y2) in self.battery_regions.items():
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.battery_selected = name
                self._draw_battery_diagram()
                return

    def _battery_state(self) -> dict:
        metrics = self.last_metrics or {
            "battery_charge": self.simulator.battery_charge,
            "dynamic_charge_power": 0.0,
            "power_mode": "--",
            "sunlight": self.simulator.is_sunlight(),
            "local_latency_ms": 0.0,
        }
        soc = float(metrics.get("battery_charge", self.simulator.battery_charge))
        charge_power = float(metrics.get("dynamic_charge_power", 0.0))
        sunlight = bool(metrics.get("sunlight", True))
        signed_power = float(metrics.get("battery_signed_power", charge_power if sunlight else -float(metrics.get("base_load", 32_000.0)) * 0.58))
        nominal_voltage = 760.0 if self.simulator.environment_mode == "space" else 420.0
        fallback_voltage = nominal_voltage * (0.88 + 0.22 * soc / 100.0)
        pack_voltage = float(metrics.get("battery_pack_voltage", fallback_voltage))
        pack_current = float(metrics.get("battery_pack_current", signed_power / max(pack_voltage, 1.0)))
        heat_w = abs(signed_power) * 0.018 + abs(pack_current) * 0.9
        pack_temp = float(metrics.get("battery_pack_temp", 22.0 + heat_w / 950.0 + (soc - 55.0) * 0.025))
        cycle_life = max(5000, int(18000 - max(0.0, pack_temp - 25.0) * 120 - abs(pack_current) * 8))
        thermal = "radiator loop active" if pack_temp > 29 or abs(pack_current) > 35 else "heater trim" if pack_temp < 15 else "passive stable"
        return {
            "soc": soc,
            "signed_power": signed_power,
            "pack_voltage": pack_voltage,
            "pack_current": pack_current,
            "pack_temp": pack_temp,
            "cycle_life": cycle_life,
            "thermal": thermal,
            "mode": metrics.get("power_mode", "--"),
            "latency": float(metrics.get("local_latency_ms", 0.0)),
            "charging": signed_power >= 0,
            "generation": float(metrics.get("total_power", 0.0)),
            "dynamic_charge_power": float(metrics.get("dynamic_charge_power", signed_power if signed_power > 0 else 0.0)),
            "fixed_charge_power": float(metrics.get("fixed_charge_power", 0.0)),
            "payload_power": float(metrics.get("payload_power", 0.0)),
            "science_headroom": float(metrics.get("science_headroom", 0.0)),
        }

    def _draw_battery_diagram(self) -> None:
        if self.battery_canvas is None or not self.battery_canvas.winfo_exists():
            return
        canvas = self.battery_canvas
        state = self._battery_state()
        width = max(canvas.winfo_width(), 760)
        height = max(canvas.winfo_height(), 600)
        canvas.delete("all")
        self.battery_regions = {}

        canvas.create_text(
            24,
            22,
            anchor="nw",
            text="Silicon Solid-State Battery Bank",
            fill=COLORS["text"],
            font=("Segoe UI", 20, "bold"),
        )
        canvas.create_text(
            24,
            54,
            anchor="nw",
            text="Concept next-generation pack: solid electrolyte + silicon anode + BMS + thermal control",
            fill=COLORS["muted"],
            font=("Segoe UI", 10),
        )
        canvas.create_rectangle(width - 290, 18, width - 28, 86, fill="#0b111a", outline=COLORS["line"])
        canvas.create_text(width - 278, 28, anchor="nw", text="SYNCED LIVE POWER", fill=COLORS["muted"], font=("Segoe UI", 8, "bold"))
        canvas.create_text(
            width - 278,
            48,
            anchor="nw",
            text=(
                f"Gen {state['generation'] / 1000:.1f} kW   "
                f"Charge {state['dynamic_charge_power'] / 1000:.1f} kW   "
                f"Payload {state['payload_power'] / 1000:.1f} kW"
            ),
            fill=COLORS["text"],
            font=("Segoe UI", 9, "bold"),
            width=245,
        )

        self._draw_clickable_box(canvas, "pv_input", 34, 105, 190, 170, f"Solar Array\n{state['generation'] / 1000:.1f} kW", COLORS["cyan"])
        self._draw_clickable_box(canvas, "bcdu", 250, 96, 430, 180, f"BCDU / MPPT\n{state['dynamic_charge_power'] / 1000:.1f} kW charge", COLORS["yellow"])
        self._draw_clickable_box(canvas, "dc_bus", width - 220, 105, width - 45, 170, f"Station DC Bus\n{state['payload_power'] / 1000:.1f} kW payload", COLORS["green"])
        canvas.create_line(190, 137, 250, 137, fill=COLORS["cyan"], width=3, arrow=tk.LAST)
        canvas.create_line(430, 137, width - 220, 137, fill=COLORS["green"], width=3, arrow=tk.LAST)

        pack_x1, pack_y1 = 70, 235
        pack_x2, pack_y2 = width - 90, height - 95
        self._draw_clickable_box(canvas, "pack", pack_x1, pack_y1, pack_x2, pack_y2, "", COLORS["line"], fill=COLORS["bg"])
        canvas.create_text(pack_x1 + 18, pack_y1 + 14, anchor="nw", text="Battery ORU Pack", fill=COLORS["text"], font=("Segoe UI", 15, "bold"))
        canvas.create_text(pack_x2 - 18, pack_y1 + 16, anchor="ne", text=f"{state['soc']:.1f}% SOC", fill=COLORS["green"], font=("Segoe UI", 16, "bold"))

        module_gap = 14
        module_w = (pack_x2 - pack_x1 - 72 - module_gap * 3) / 4
        module_y1 = pack_y1 + 70
        module_y2 = pack_y2 - 42
        for index in range(4):
            x1 = int(pack_x1 + 28 + index * (module_w + module_gap))
            x2 = int(x1 + module_w)
            name = f"module_{index + 1}"
            color = ARRAY_COLORS[index]
            self._draw_clickable_box(canvas, name, x1, module_y1, x2, module_y2, "", color, fill=COLORS["panel_alt"])
            canvas.create_text((x1 + x2) // 2, module_y1 + 18, text=f"Module {index + 1}", fill=color, font=("Segoe UI", 10, "bold"))
            self._draw_cell_cutaway(canvas, x1 + 12, module_y1 + 46, x2 - 12, module_y2 - 20, state, index)

        lower_y = height - 70
        self._draw_clickable_box(canvas, "thermal", 62, lower_y, 230, lower_y + 42, "Thermal Loop", COLORS["blue"])
        self._draw_clickable_box(canvas, "bms", 260, lower_y, 445, lower_y + 42, "BMS + Fault Logic", COLORS["yellow"])
        self._draw_clickable_box(canvas, "safety", 475, lower_y, 660, lower_y + 42, "Isolation + Contactors", COLORS["red"])

        flow_color = COLORS["green"] if state["charging"] else COLORS["blue"]
        flow_text = "Li+ to silicon anode (charging)" if state["charging"] else "Li+ to cathode (discharging)"
        canvas.create_line(430, 180, width // 2, pack_y1, fill=flow_color, width=3, arrow=tk.LAST)
        canvas.create_text(width // 2 + 8, pack_y1 - 22, anchor="w", text=flow_text, fill=flow_color, font=("Segoe UI", 10, "bold"))

        self._update_battery_side_panel(state)

    def _draw_cell_cutaway(self, canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int, state: dict, index: int) -> None:
        height = y2 - y1
        layers = [
            ("cathode", "Cathode\nNMC/LFP", "#4f8cff", 0.27),
            ("electrolyte", "Solid\nElectrolyte", "#f2cc4d", 0.18),
            ("anode", "Silicon\nAnode", "#31d17c", 0.27),
        ]
        cursor = y1
        for name, label, color, fraction in layers:
            layer_h = int(height * fraction)
            region = f"{name}_{index + 1}"
            self._draw_clickable_box(canvas, region, x1, cursor, x2, cursor + layer_h, label, color, fill=color)
            cursor += layer_h + 5

        ion_color = COLORS["text"]
        direction = -1 if state["charging"] else 1
        for offset in range(6):
            x = x1 + 16 + offset * max(12, (x2 - x1 - 32) // 6)
            y_start = y2 - 22 if state["charging"] else y1 + 22
            y_end = y1 + 32 if state["charging"] else y2 - 32
            canvas.create_line(x, y_start, x, y_end, fill=ion_color, dash=(3, 3), arrow=tk.LAST if direction else None)
            canvas.create_oval(x - 3, y_start - 3, x + 3, y_start + 3, fill=ion_color, outline="")

    def _draw_clickable_box(
        self,
        canvas: tk.Canvas,
        name: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        text: str,
        outline: str,
        fill: str | None = None,
    ) -> None:
        fill = fill or COLORS["panel_alt"]
        width = 4 if self.battery_selected == name or self.battery_selected in name else 2
        canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=width)
        if text:
            canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text=text, fill=COLORS["text"], font=("Segoe UI", 9, "bold"), width=max(70, x2 - x1 - 12))
        self.battery_regions[name] = (x1, y1, x2, y2)

    def _update_battery_side_panel(self, state: dict) -> None:
        if self.battery_metric_labels:
            self.battery_metric_labels["soc"].configure(text=f"{state['soc']:.1f}%")
            self.battery_metric_labels["pack_voltage"].configure(text=f"{state['pack_voltage']:.0f} V")
            self.battery_metric_labels["pack_current"].configure(text=f"{state['pack_current']:+.1f} A")
            self.battery_metric_labels["pack_power"].configure(text=f"{state['signed_power'] / 1000:+.1f} kW")
            self.battery_metric_labels["pack_temp"].configure(text=f"{state['pack_temp']:.1f} C")
            self.battery_metric_labels["cycle_life"].configure(text=f"{state['cycle_life']:,} cycles")
            self.battery_metric_labels["thermal"].configure(text=state["thermal"])
            self.battery_metric_labels["latency"].configure(text=f"{state['latency']:.2f} ms | fixed rule {state['fixed_charge_power'] / 1000:.1f} kW")

        if self.battery_detail_label is None:
            return
        details = {
            "overview": (
                "This diagram combines station power architecture with a cell cutaway. Solar arrays feed a charge/discharge controller, "
                "which either charges the pack during sunlight or lets the pack support the DC bus during eclipse. The pack is drawn as "
                "replaceable modules with BMS sensing, isolation contactors, and thermal control."
            ),
            "pv_input": "Solar arrays provide DC power during sunlight. The controller compares live generation, base station load, payload target, and battery state before deciding how much excess power to store.",
            "bcdu": "The charge/discharge controller acts like an ISS-style BCDU plus MPPT logic: it limits current, prevents overcharge, and reverses flow during eclipse so the battery can hold up the station bus.",
            "dc_bus": "The regulated DC bus feeds life support, avionics, communications, and payload racks. In this concept the controller preserves payload power before sending surplus to storage.",
            "pack": "The pack is organized as serviceable modules. Series strings raise voltage; parallel strings increase capacity and redundancy. The BMS watches voltage spread, temperature, isolation, and current.",
            "thermal": "Space batteries need active temperature management. During sunlight, waste heat is rejected to radiators; during eclipse, heaters may keep cells in their preferred operating band.",
            "bms": "The BMS estimates state of charge and health, balances cells, catches sensor disagreement, and opens contactors if current, temperature, isolation, or vibration goes out of bounds.",
            "safety": "Isolation monitoring and contactors let the pack disconnect from the station bus during faults. This is the fallback path that keeps the new intelligence additive, not a single point of failure.",
        }
        if self.battery_selected.startswith("cathode"):
            text = "Cathode composite: the positive electrode stores lithium when the pack is discharged. During charging, lithium ions leave the cathode and move through the solid electrolyte."
        elif self.battery_selected.startswith("electrolyte"):
            text = "Solid electrolyte: replaces flammable liquid electrolyte and separator. It conducts lithium ions while keeping anode and cathode electronically separated."
        elif self.battery_selected.startswith("anode"):
            text = "Silicon anode: stores lithium as lithium-silicon alloy during charge. Silicon can store far more lithium than graphite, but it expands, so pressure, interfaces, and solid-electrolyte contact matter."
        elif self.battery_selected.startswith("module_"):
            text = "Battery module: a stack of cells with local temperature, voltage, and pressure sensing. Modules are balanced so one weak string does not silently reduce available eclipse energy."
        else:
            text = details.get(self.battery_selected, details["overview"])
        self.battery_detail_label.configure(text=text)

    def _on_system_flow_click(self, event) -> None:
        for name, (x1, y1, x2, y2) in self.system_flow_regions.items():
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.system_flow_selected = name
                self._draw_system_flow_diagram()
                return

    def _system_flow_state(self) -> dict:
        metrics = self.last_metrics or {
            "total_power": 0.0,
            "dynamic_charge_power": 0.0,
            "battery_charge": self.simulator.battery_charge,
            "payload_power": 0.0,
            "science_headroom": 0.0,
            "power_mode": "--",
            "sunlight": self.simulator.is_sunlight(),
            "system_loads": {},
            "power_margin": 0.0,
        }
        generation = float(metrics.get("total_power", 0.0))
        battery_power = float(metrics.get("dynamic_charge_power", 0.0))
        if not metrics.get("sunlight", True):
            battery_power = -float(metrics.get("base_load", 32_000.0)) * 0.58
        payload_power = float(metrics.get("payload_power", 0.0))
        loads = dict(metrics.get("system_loads", {}))
        critical_power = sum(
            loads.get(key, 0.0)
            for key in ("systems_deck", "crew_hab", "utility_bay", "thermal", "propulsion", "docking_robotics")
        )
        used = sum(value for key, value in loads.items() if key != "unallocated_margin") + max(battery_power, 0.0)
        margin = float(metrics.get("power_margin", max(0.0, generation - used)))
        return {
            "generation": generation,
            "battery_power": battery_power,
            "payload_power": payload_power,
            "critical_power": critical_power,
            "loads": loads,
            "margin": margin,
            "soc": float(metrics.get("battery_charge", self.simulator.battery_charge)),
            "mode": metrics.get("power_mode", "--"),
            "sunlight": bool(metrics.get("sunlight", True)),
            "environment_mode": metrics.get("environment_mode", "space"),
            "fixed_charge_power": float(metrics.get("fixed_charge_power", 0.0)),
            "dynamic_charge_power": float(metrics.get("dynamic_charge_power", 0.0)),
            "science_headroom": float(metrics.get("science_headroom", payload_power + margin)),
        }

    def _draw_system_flow_diagram(self) -> None:
        if self.system_flow_canvas is None or not self.system_flow_canvas.winfo_exists():
            return
        canvas = self.system_flow_canvas
        state = self._system_flow_state()
        width = max(canvas.winfo_width(), 820)
        height = max(canvas.winfo_height(), 620)
        canvas.delete("all")
        self.system_flow_regions = {}

        canvas.create_text(
            24,
            20,
            anchor="nw",
            text="Starlab Energy Node Map",
            fill=COLORS["text"],
            font=("Segoe UI", 20, "bold"),
        )
        totals_x1 = max(24, width - 285)
        canvas.create_rectangle(totals_x1, 18, width - 28, 82, fill="#0b111a", outline=COLORS["line"])
        canvas.create_text(totals_x1 + 12, 28, anchor="nw", text="LIVE TOTALS", fill=COLORS["muted"], font=("Segoe UI", 8, "bold"))
        canvas.create_text(
            totals_x1 + 12,
            47,
            anchor="nw",
            text=(
                f"Gen {state['generation'] / 1000:.1f} kW   "
                f"Batt {state['battery_power'] / 1000:+.1f} kW   "
                f"Free {state['margin'] / 1000:.1f} kW"
            ),
            fill=COLORS["text"],
            font=("Segoe UI", 9, "bold"),
            width=max(170, width - totals_x1 - 28),
        )
        canvas.create_text(
            24,
            54,
            anchor="nw",
            text="Cutaway map: solar wings and service module feed a central power spine through the Utility Bay, Systems Deck, Payload Lab, Crew Habitation Deck, docking/airlock interfaces, and external payload platforms.",
            fill=COLORS["muted"],
            font=("Segoe UI", 10),
            width=max(360, totals_x1 - 58),
            tags=("flow_label",),
        )

        # Orientation labels are deliberately station-map style. This is a public-layout
        # reconstruction, not a copied engineering drawing.
        cx = width // 2
        top = 112
        bottom = height - 212
        hab_w = min(300, max(240, width // 3))
        hab_x1 = cx - hab_w // 2
        hab_x2 = cx + hab_w // 2
        hab_h = bottom - top
        deck_h = hab_h / 7.0

        canvas.create_text(hab_x1 - 22, top + 6, anchor="ne", text="ZENITH", fill=COLORS["muted"], font=("Segoe UI", 8, "bold"), tags=("flow_label",))
        canvas.create_text(hab_x1 - 22, bottom - 6, anchor="se", text="NADIR / SERVICE END", fill=COLORS["muted"], font=("Segoe UI", 8, "bold"), tags=("flow_label",))
        canvas.create_text(hab_x1, bottom + 24, anchor="w", text="7.7-8 m class habitat/lab cutaway", fill=COLORS["muted"], font=("Segoe UI", 8), tags=("flow_label",))

        # Habitat cylinder silhouette with rounded pressure-shell ends.
        canvas.create_oval(hab_x1, top - 28, hab_x2, top + 38, fill="#192433", outline=COLORS["line"], width=2)
        canvas.create_rectangle(hab_x1, top + 6, hab_x2, bottom - 6, fill="#111a25", outline=COLORS["line"], width=2)
        canvas.create_oval(hab_x1, bottom - 38, hab_x2, bottom + 28, fill="#111a25", outline=COLORS["line"], width=2)

        deck_defs = [
            ("stowage_upper", "Deck 06\nStowage / overhead", COLORS["muted"], 0),
            ("stowage_mid", "Deck 05\nStowage / logistics", COLORS["muted"], 1),
            ("stowage_low", "Deck 04\nStowage / access", COLORS["muted"], 2),
            ("crew_hab", "Deck 03\nCrew Habitation", COLORS["yellow"], 3),
            ("payload_lab", "Deck 02\nInternal Payload Lab", COLORS["cyan"], 4),
            ("systems_deck", "Deck 01\nSystems Deck", COLORS["green"], 5),
            ("utility_bay", "Deck 00\nUtility Bay", COLORS["blue"], 6),
        ]
        deck_centers: dict[str, tuple[int, int]] = {}
        for name, label, color, index in deck_defs:
            y1 = int(top + index * deck_h)
            y2 = int(top + (index + 1) * deck_h)
            fill = COLORS["panel_alt"] if index % 2 else "#142030"
            selected = self.system_flow_selected == name
            canvas.create_rectangle(hab_x1 + 10, y1 + 3, hab_x2 - 10, y2 - 3, fill=fill, outline=color, width=4 if selected else 1)
            value_kw = state["loads"].get(name, 0.0) / 1000.0
            label_y = (y1 + y2) // 2
            short_label = label.replace("Stowage / ", "").replace("Internal ", "").replace("Crew ", "")
            canvas.create_text(hab_x1 - 12, label_y, anchor="e", text=short_label.replace("\n", " | "), fill=COLORS["text"], font=("Segoe UI", 8, "bold"), width=128, justify="right", tags=("flow_label",))
            self._value_badge(canvas, hab_x2 + 14, label_y, f"{value_kw:.1f} kW", color, anchor="w")
            self.system_flow_regions[name] = (hab_x1 + 10, y1 + 3, hab_x2 - 10, y2 - 3)
            deck_centers[name] = (cx, (y1 + y2) // 2)

        # Central hatch/power spine, aligned with the published stacked-deck concept.
        spine_x = cx + hab_w // 2 - 42
        canvas.create_line(spine_x, top + 18, spine_x, bottom - 18, fill=COLORS["green"], width=5)
        self._value_badge(canvas, spine_x + 18, top + 34, f"{state['critical_power'] / 1000:.1f} kW spine", COLORS["green"], anchor="w")
        self.system_flow_regions["dc_bus"] = (spine_x - 12, top + 18, spine_x + 12, bottom - 18)

        # Service module and solar wings at the nadir/service end.
        service_y1 = bottom + 16
        service_y2 = min(height - 118, bottom + 82)
        service_x1 = cx - 118
        service_x2 = cx + 118
        self._map_node(canvas, "service_module", service_x1, service_y1, service_x2, service_y2, "Service Module", COLORS["yellow"], f"{state['generation'] / 1000:.1f} kW in")
        self._map_node(canvas, "battery_bank", hab_x1 + 24, bottom - 70, hab_x1 + 140, bottom - 18, "Battery", COLORS["blue"], f"{state['soc']:.0f}% | {state['battery_power'] / 1000:+.1f} kW")

        wing_y = (service_y1 + service_y2) // 2
        left_wing_root = service_x1
        right_wing_root = service_x2
        left_tip = max(34, service_x1 - 260)
        right_tip = min(width - 34, service_x2 + 260)
        self._draw_solar_wing(canvas, left_tip, wing_y, left_wing_root, wing_y, reverse=True, array_ids=(1, 2))
        self._draw_solar_wing(canvas, right_wing_root, wing_y, right_tip, wing_y, reverse=False, array_ids=(3, 4))
        self.system_flow_regions["solar_arrays"] = (left_tip, wing_y - 66, right_tip, wing_y + 66)
        if left_tip > 190:
            self._value_badge(canvas, left_tip - 14, wing_y - 48, f"4 arrays total {state['generation'] / 1000:.1f} kW", COLORS["cyan"], anchor="e")
        else:
            self._value_badge(canvas, 28, wing_y - 62, f"4 arrays total {state['generation'] / 1000:.1f} kW", COLORS["cyan"], anchor="w")

        # Docking ports, airlock, external platforms, robotic arm, and thermal radiators.
        dock_y = deck_centers["stowage_mid"][1]
        self._map_node(
            canvas,
            "docking_robotics",
            hab_x2 + 86,
            dock_y - 28,
            hab_x2 + 238,
            dock_y + 28,
            "Docking +\nRobotics",
            COLORS["red"],
            f"{state['loads'].get('docking_robotics', 0) / 1000:.1f} kW",
        )

        payload_y = deck_centers["payload_lab"][1]
        self._map_node(canvas, "external_payloads", hab_x2 + 72, payload_y - 54, hab_x2 + 238, payload_y - 2, "External Payload\nPlatforms", COLORS["cyan"], f"{state['loads'].get('external_payloads', 0) / 1000:.1f} kW")
        self._map_node(canvas, "airlock", hab_x2 + 72, payload_y + 18, hab_x2 + 238, payload_y + 70, "Bishop Airlock\npayload transfer", COLORS["yellow"], f"{state['loads'].get('airlock', 0) / 1000:.1f} kW")
        arm_x1 = hab_x2 + 22
        arm_y1 = deck_centers["crew_hab"][1] - 8
        canvas.create_line(arm_x1, arm_y1, arm_x1 + 72, arm_y1 - 46, arm_x1 + 154, arm_y1 - 32, fill=COLORS["red"], width=4, smooth=True)
        canvas.create_oval(arm_x1 + 148, arm_y1 - 38, arm_x1 + 162, arm_y1 - 24, fill=COLORS["red"], outline="")
        self._value_badge(canvas, arm_x1 + 76, arm_y1 - 68, "SKYMAKER arm", COLORS["red"])

        thermal_y = deck_centers["systems_deck"][1] + 36
        thermal_x1 = max(34, left_tip + 18)
        thermal_x2 = thermal_x1 + 134
        self._draw_radiator(canvas, thermal_x1, thermal_y - 56, thermal_x2, thermal_y + 56)
        self.system_flow_regions["thermal"] = (thermal_x1 - 10, thermal_y - 62, thermal_x2 + 8, thermal_y + 62)

        propulsion_y = service_y2 - 12
        canvas.create_polygon(service_x1 + 28, service_y2, service_x1 + 56, service_y2, service_x1 + 42, service_y2 + 26, fill=COLORS["yellow"], outline=COLORS["yellow"])
        canvas.create_polygon(service_x2 - 56, service_y2, service_x2 - 28, service_y2, service_x2 - 42, service_y2 + 26, fill=COLORS["yellow"], outline=COLORS["yellow"])
        self.system_flow_regions["propulsion"] = (service_x1, service_y1, service_x2, service_y2 + 32)

        # Power paths are overlaid on the physical map instead of replacing it.
        self._power_line(canvas, left_wing_root, wing_y, service_x1 + 28, wing_y, COLORS["cyan"], state["generation"] / 2)
        self._power_line(canvas, right_wing_root, wing_y, service_x2 - 28, wing_y, COLORS["cyan"], state["generation"] / 2)
        self._power_line(canvas, cx, service_y1, spine_x, bottom - 18, COLORS["green"], max(state["critical_power"], 1.0))
        self._power_line(canvas, hab_x1 + 132, bottom - 44, spine_x, bottom - 30, COLORS["blue"], abs(state["battery_power"]))

        load_links = [
            ("stowage_upper", COLORS["muted"], state["loads"].get("stowage_upper", 0)),
            ("stowage_mid", COLORS["muted"], state["loads"].get("stowage_mid", 0)),
            ("stowage_low", COLORS["muted"], state["loads"].get("stowage_low", 0)),
            ("systems_deck", COLORS["green"], state["loads"].get("systems_deck", 0)),
            ("payload_lab", COLORS["cyan"], state["loads"].get("payload_lab", 0)),
            ("crew_hab", COLORS["yellow"], state["loads"].get("crew_hab", 0)),
            ("utility_bay", COLORS["blue"], state["loads"].get("utility_bay", 0)),
            ("external_payloads", COLORS["cyan"], state["loads"].get("external_payloads", 0)),
            ("airlock", COLORS["yellow"], state["loads"].get("airlock", 0)),
            ("thermal", COLORS["blue"], state["loads"].get("thermal", 0)),
            ("docking_robotics", COLORS["red"], state["loads"].get("docking_robotics", 0)),
        ]
        for name, color, watts in load_links:
            target = self._region_center(self.system_flow_regions[name])
            self._power_line(canvas, spine_x, target[1], target[0], target[1], color, watts)

        mode_color = COLORS["yellow"] if state["sunlight"] else COLORS["blue"]
        canvas.create_oval(28, height - 86, 88, height - 26, fill=mode_color, outline="")
        canvas.create_text(58, height - 56, text="SUN" if state["sunlight"] else "ECL", fill=COLORS["bg"], font=("Segoe UI", 10, "bold"))
        canvas.create_text(104, height - 64, anchor="w", text=f"Mode: {state['mode']}", fill=COLORS["text"], font=("Segoe UI", 11, "bold"))
        canvas.create_text(104, height - 42, anchor="w", text=f"Free margin: {state['margin'] / 1000:.1f} kW | science allocation: {state['science_headroom'] / 1000:.1f} kW", fill=COLORS["muted"], font=("Segoe UI", 10))

        self._update_system_flow_side_panel(state)

    def _region_center(self, region: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = region
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _map_node(
        self,
        canvas: tk.Canvas,
        name: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        color: str,
        value: str,
    ) -> None:
        selected = self.system_flow_selected == name
        canvas.create_rectangle(x1, y1, x2, y2, fill=COLORS["panel_alt"], outline=color, width=4 if selected else 2)
        canvas.create_text(x1 + 8, y1 + 7, anchor="nw", text=label, fill=COLORS["text"], font=("Segoe UI", 8, "bold"), width=x2 - x1 - 16, tags=("flow_label",))
        canvas.create_text(x2 - 8, y2 - 8, anchor="se", text=value, fill=color, font=("Segoe UI", 8, "bold"), width=x2 - x1 - 16, tags=("flow_label",))
        self.system_flow_regions[name] = (x1, y1, x2, y2)

    def _value_badge(self, canvas: tk.Canvas, x: int, y: int, text: str, color: str, anchor: str = "center") -> None:
        font = ("Segoe UI", 8, "bold")
        width = min(150, max(54, len(text) * 7))
        height = 20
        if anchor == "e":
            x1, x2 = x - width, x
        elif anchor == "w":
            x1, x2 = x, x + width
        else:
            x1, x2 = x - width // 2, x + width // 2
        y1, y2 = y - height // 2, y + height // 2
        canvas.create_rectangle(x1, y1, x2, y2, fill="#071018", outline=color, tags=("flow_label",))
        canvas.create_text((x1 + x2) // 2, y, text=text, fill=color, font=font, width=max(48, width - 8), tags=("flow_label",))

    def _draw_solar_wing(
        self,
        canvas: tk.Canvas,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        reverse: bool,
        array_ids: tuple[int, int],
    ) -> None:
        span = abs(x2 - x1)
        panel_count = 6
        panel_w = max(22, span // panel_count)
        start = min(x1, x2)
        array_power = {
            reading.array_id: reading.power
            for reading in self.latest_readings
        }
        midpoint = start + span // 2
        for index in range(panel_count):
            px1 = start + index * panel_w + 4
            px2 = min(start + (index + 1) * panel_w - 4, max(x1, x2))
            array_id = array_ids[0] if (px1 + px2) // 2 < midpoint else array_ids[1]
            canvas.create_rectangle(px1, y1 - 38, px2, y1 + 38, fill="#12313d", outline=COLORS["cyan"], width=1)
            canvas.create_line(px1, y1, px2, y1, fill=COLORS["cyan"])
            canvas.create_line((px1 + px2) // 2, y1 - 38, (px1 + px2) // 2, y1 + 38, fill=COLORS["line"])
        canvas.create_line(x1, y1, x2, y2, fill=COLORS["cyan"], width=3)
        for index, array_id in enumerate(array_ids):
            center_x = start + int(span * (0.25 if index == 0 else 0.75))
            power_kw = array_power.get(array_id, 0.0) / 1000.0
            self._value_badge(canvas, center_x, y1 + (52 if y1 < (self.system_flow_canvas.winfo_height() // 2 if self.system_flow_canvas else 0) else -52), f"A{array_id} {power_kw:.1f} kW", ARRAY_COLORS[array_id - 1])

    def _draw_radiator(self, canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int) -> None:
        canvas.create_polygon(x1, y1 + 18, x2, y1, x2, y2 - 18, x1, y2, fill="#122438", outline=COLORS["blue"], width=2)
        for offset in range(12, max(13, x2 - x1), 18):
            canvas.create_line(x1 + offset, y1 + 14, x1 + offset - 18, y2 - 6, fill=COLORS["line"])
        if x1 > 118:
            canvas.create_text(x1 - 12, (y1 + y2) // 2, anchor="e", text="thermal\nradiators", fill=COLORS["blue"], font=("Segoe UI", 8, "bold"), justify="right", tags=("flow_label",))
        else:
            canvas.create_text(x1 + 8, y1 + 6, anchor="nw", text="thermal\nradiators", fill=COLORS["blue"], font=("Segoe UI", 8, "bold"), tags=("flow_label",))

    def _flow_box(
        self,
        canvas: tk.Canvas,
        name: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        color: str,
        value: str,
    ) -> None:
        selected = self.system_flow_selected == name
        canvas.create_rectangle(x1, y1, x2, y2, fill=COLORS["panel_alt"], outline=color, width=4 if selected else 2)
        canvas.create_text(x1 + 10, y1 + 10, anchor="nw", text=label, fill=COLORS["text"], font=("Segoe UI", 9, "bold"), width=x2 - x1 - 20)
        canvas.create_text(x2 - 10, y2 - 12, anchor="se", text=value, fill=color, font=("Segoe UI", 10, "bold"))
        self.system_flow_regions[name] = (x1, y1, x2, y2)

    def _power_line(self, canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int, color: str, watts: float) -> None:
        if watts <= 1:
            line_id = canvas.create_line(x1, y1, x2, y2, fill=COLORS["line"], width=1, dash=(4, 4))
            if canvas.find_withtag("flow_label"):
                canvas.tag_lower(line_id, "flow_label")
            return
        width = max(2, min(7, int(watts / 6000)))
        line_id = canvas.create_line(x1, y1, x2, y2, fill=color, width=width, arrow=tk.LAST)
        phase = (time.time() * 0.65) % 1.0
        dot_x = x1 + (x2 - x1) * phase
        dot_y = y1 + (y2 - y1) * phase
        dot_id = canvas.create_oval(dot_x - 4, dot_y - 4, dot_x + 4, dot_y + 4, fill=color, outline="")
        if canvas.find_withtag("flow_label"):
            canvas.tag_lower(line_id, "flow_label")
            canvas.tag_lower(dot_id, "flow_label")

    def _update_system_flow_side_panel(self, state: dict) -> None:
        if self.system_flow_metric_labels:
            self.system_flow_metric_labels["generation"].configure(text=f"{state['generation'] / 1000:.1f} kW")
            self.system_flow_metric_labels["battery"].configure(text=f"{state['battery_power'] / 1000:+.1f} kW")
            self.system_flow_metric_labels["dc_bus"].configure(text=f"{state['critical_power'] / 1000:.1f} kW")
            self.system_flow_metric_labels["payload"].configure(text=f"{state['payload_power'] / 1000:.1f} / {STARLAB_PAYLOAD_POWER_W / 1000:.0f} kW")
            critical_sum = sum(
                state["loads"].get(key, 0.0)
                for key in ("systems_deck", "crew_hab", "utility_bay", "thermal", "propulsion", "docking_robotics")
            )
            self.system_flow_metric_labels["critical"].configure(text=f"{critical_sum / 1000:.1f} kW")
            self.system_flow_metric_labels["thermal"].configure(text=f"{state['loads'].get('thermal', 0) / 1000:.1f} kW")
            self.system_flow_metric_labels["propulsion"].configure(text=f"{state['loads'].get('propulsion', 0) / 1000:.1f} kW")
            self.system_flow_metric_labels["margin"].configure(text=f"{state['margin'] / 1000:.1f} kW")

        if self.system_flow_detail_label is None:
            return
        details = {
            "overview": "This whole-station view follows public Starlab architecture: solar power enters the service module, is conditioned onto a station DC bus, charges batteries for eclipse, and feeds the habitat/lab module decks plus external payload services.",
            "solar_arrays": "Solar array wings provide the primary source of energy. The monitoring model receives per-channel voltage, current, exposure, temperature, vibration, and benchmark data before the controller decides routing.",
            "service_module": "The service module is the power and propulsion end of Starlab. In this diagram it hosts power conditioning, charge/discharge control, propulsion/GNC reserves, and the main switching path into the pressurized station.",
            "dc_bus": "The regulated DC bus distributes conditioned power. Critical station loads receive priority, then payloads, then surplus battery charging and science headroom.",
            "battery_bank": "The battery bank stores surplus power during sunlight and discharges through the bus during orbital shadow. The battery tab shows the cell-level solid-state concept.",
            "stowage_upper": "Upper stowage decks are modeled from Starlab's published stacked-deck layout. They carry logistics, spares, soft goods, and overhead utility distribution above the main crew and lab decks.",
            "stowage_mid": "Mid stowage sits above the crew deck, close to docking and transfer routes. In the map it receives keep-alive power for lighting, sensors, inventory tracking, and ventilation.",
            "stowage_low": "Lower stowage is a buffer between the crew area and payload laboratory, useful for experiment support gear and fast-access maintenance hardware.",
            "systems_deck": "The Systems Deck contains the station control backbone: avionics, command/data handling, ECLSS controls, power switching, monitoring, and fault management.",
            "payload_lab": "The Internal Payload Laboratory is modeled as the largest flexible science load. Public Starlab payload information emphasizes power, cooling, nitrogen, and vacuum support for experiments.",
            "crew_hab": "The Crew Habitation Deck includes crew quarters and daily living systems, including lighting, galley equipment, medical gear, displays, and local ventilation.",
            "utility_bay": "The Utility Bay carries support functions such as water, oxygen, pumps, spares, maintenance interfaces, and equipment that keeps the habitat serviceable.",
            "external_payloads": "External payload interfaces receive power and thermal services for exposed experiments mounted outside the pressure vessel.",
            "airlock": "The Bishop Airlock is shown adjacent to the payload region because it supports transfer between internal experiments and external platforms.",
            "thermal": "Thermal heat rejection converts electrical heat load into radiator demand. Payload racks, batteries, avionics, and life support all drive this loop.",
            "propulsion": "Propulsion and guidance/navigation/control loads support attitude control, momentum management, and reboost reserves from the service module.",
            "docking_robotics": "Docking, cargo transfer, and robotics loads cover visiting vehicle support and external payload handling when operations require it.",
        }
        self.system_flow_detail_label.configure(text=details.get(self.system_flow_selected, details["overview"]))


    def _play_alert_sound(self) -> None:
        try:
            if self._sound:
                self._sound.play()
        except Exception:
            pass

    def _load_model_async(self) -> None:
        def worker() -> None:
            try:
                self.model.load()
                self.model_status = (
                    f"Loaded {self.model.model_path}\n"
                    f"Validation accuracy: {self.model.accuracy:.2%}\n"
                    f"Runtime: {cuda_summary()}"
                )
                self.model_ready = True
            except Exception as exc:
                self.model_status = f"Model load failed: {exc}"
            self.ui_queue.put("model")

        threading.Thread(target=worker, daemon=True).start()

    def _retrain_async(self) -> None:
        self.model_ready = False
        self.model_status = "Retraining model. CUDA will be used if available..."
        self.ui_queue.put("model")

        def worker() -> None:
            try:
                result = train_and_save(samples=52000, epochs=28)
                self.model = StarlabAnomalyModel()
                self.model.load()
                self.model_ready = True
                self.model_status = (
                    f"Retrained on {result.device}\n"
                    f"Saved: {result.model_path}\n"
                    f"Validation accuracy: {result.accuracy:.2%}"
                )
            except Exception as exc:
                self.model_status = f"Retraining failed: {exc}"
            self.ui_queue.put("model")

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")

    def _select_battery_tab(self) -> None:
        if self.main_tabs is not None and self.battery_tab is not None:
            self.main_tabs.select(self.battery_tab)
        self._draw_battery_diagram()

    def _select_system_flow_tab(self) -> None:
        if self.main_tabs is not None and self.system_flow_tab is not None:
            self.main_tabs.select(self.system_flow_tab)
        self._draw_system_flow_diagram()

    def _select_benchmark_tab(self) -> None:
        if self.main_tabs is not None and self.benchmark_tab is not None:
            self.main_tabs.select(self.benchmark_tab)

    def _toggle_test_mode(self) -> None:
        self.simulator.test_mode = self.test_var.get()

    def _change_environment(self) -> None:
        mode = "earth" if self.environment_var.get() == "Earth Solar Farm" else "space"
        self.simulator.set_environment_mode(mode)
        self.orbit_var.set("auto")
        self.power_history.clear()
        self.efficiency_history.clear()
        self.latest_readings = []
        self.latest_predictions.clear()
        self.last_incident.clear()
        self.environment_label.configure(text="Earth" if mode == "earth" else "Space")
        for array_id in self.array_labels:
            self._populate_array_actions(array_id)

    def _acknowledge_alerts(self) -> None:
        acknowledge_all()
        self.alert_label.configure(text="Alerts acknowledged", fg=COLORS["green"])
        self._refresh_database_views()

    def _tick(self) -> None:
        self._drain_queue()
        if not self.paused:
            self._sample_and_update()
        self.root.after(1000, self._tick)

    def _drain_queue(self) -> None:
        while True:
            try:
                message = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if message == "model":
                self.model_label.configure(text=self.model_status)
                self.device_label.configure(text=cuda_summary())

    def _sample_and_update(self) -> None:
        dt = float(self.sim_speed.get())
        readings, metrics = self.simulator.next_readings(dt_seconds=dt)
        self.latest_readings = readings
        predicted_rows = []
        active_alerts = []
        prediction_latency_ms = 0.0

        for reading in readings:
            if self.model_ready:
                started = time.perf_counter()
                predicted, confidence, _scores = self.model.predict(reading.features())
                prediction_latency_ms += (time.perf_counter() - started) * 1000
            else:
                predicted, confidence = "MODEL_LOADING", 0.0
            self.latest_predictions[reading.array_id] = (predicted, confidence)
            predicted_rows.append(
                reading.as_database_row(
                    predicted_label=predicted,
                    confidence=confidence,
                    battery_charge=metrics["battery_charge"],
                )
            )

            incident_label = self._incident_label(reading, predicted, confidence)
            if incident_label:
                active_alerts.append((reading.array_id, incident_label, confidence))
                self._record_incident(reading.array_id, incident_label, confidence)
                if metrics["environment_mode"] == "space" and incident_label in IMPACT_CLASSES:
                    self.benchmark_avoided_eva_events.add((reading.array_id, incident_label))

        if readings:
            metrics["local_latency_ms"] = prediction_latency_ms / len(readings)
        else:
            metrics["local_latency_ms"] = 0.0
        self._update_benchmark_accounting(metrics, dt)
        self.last_metrics = metrics.copy()

        insert_readings(predicted_rows)
        insert_metric(
            {
                "timestamp": metrics["timestamp"],
                "total_power": metrics["total_power"],
                "average_efficiency": metrics["average_efficiency"],
                "total_energy_wh": metrics["total_energy_wh"],
                "battery_charge": metrics["battery_charge"],
                "power_mode": metrics["power_mode"],
                "excess_power": metrics["excess_power"],
                "cuda_enabled": 1 if self.model.cuda_enabled else 0,
                "environment_mode": metrics["environment_mode"],
                "fixed_charge_power": metrics["fixed_charge_power"],
                "dynamic_charge_power": metrics["dynamic_charge_power"],
                "battery_priority_fraction": metrics["battery_priority_fraction"],
                "payload_power": metrics["payload_power"],
                "science_headroom": metrics["science_headroom"],
                "power_margin": metrics["power_margin"],
                "battery_signed_power": metrics["battery_signed_power"],
                "battery_pack_voltage": metrics["battery_pack_voltage"],
                "battery_pack_current": metrics["battery_pack_current"],
                "battery_pack_temp": metrics["battery_pack_temp"],
                "local_latency_ms": metrics["local_latency_ms"],
            }
        )

        self.power_history = (self.power_history + [metrics["total_power"]])[-72:]
        self.efficiency_history = (self.efficiency_history + [metrics["average_efficiency"]])[-72:]
        self._update_header(metrics, active_alerts)
        self._update_array_cards(readings)
        self._update_routing_views(metrics, readings)
        self._draw_network(metrics, active_alerts)
        self._draw_charts()
        self._draw_battery_diagram()
        self._draw_system_flow_diagram()
        self._update_benchmark_tab(metrics)
        self._refresh_database_views()

    def _incident_label(self, reading, predicted: str, confidence: float) -> str | None:
        if reading.environment_mode == "space":
            if reading.actual_label in IMPACT_CLASSES:
                return reading.actual_label
            if reading.vibration_g >= 1.0:
                return "LARGE_IMPACT"
            if reading.vibration_g >= 0.14:
                return "SMALL_IMPACT"
            if predicted == "LARGE_IMPACT":
                if confidence >= 0.86 and reading.vibration_g >= 0.45 and reading.benchmark_ratio <= 0.78:
                    return "LARGE_IMPACT"
                return None
            if predicted == "SMALL_IMPACT":
                if confidence >= 0.78 and reading.vibration_g >= 0.08 and reading.benchmark_ratio <= 0.92:
                    return "SMALL_IMPACT"
                return None
        if reading.actual_label in SEVERITY:
            return reading.actual_label
        if predicted in SEVERITY and confidence >= 0.72:
            return predicted
        return None

    def _record_incident(self, array_id: int, label: str, confidence: float) -> None:
        key = (array_id, label)
        now = time.time()
        if now - self.last_incident.get(key, 0) < 12:
            return
        self.last_incident[key] = now
        severity, message = self._severity_message(label)
        insert_incident(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "array_id": array_id,
                "incident_type": label,
                "severity": severity,
                "confidence": confidence,
                "message": message,
            }
        )
        if label in IMPACT_CLASSES:
            self._play_alert_sound()

    def _severity_message(self, label: str) -> tuple[str, str]:
        severity, space_message = SEVERITY[label]
        if self.simulator.environment_mode != "earth":
            return severity, space_message
        earth_messages = {
            "PARTIAL_SHADOW": "Tree, building, or localized obstruction is shadowing a PV string",
            "CLOUD_COVER": "Cloud cover is reducing output relative to clear-sky benchmark",
            "SOILING": "Soiling, pollen, or dust buildup is reducing terrestrial array output",
            "BENCHMARK_UNDERPERFORMANCE": "Power is below expected benchmark for current irradiance and temperature",
            "STRING_FAULT": "One PV string is producing much less current than neighboring strings",
            "INVERTER_CLIPPING": "Inverter is clipping peak DC production",
            "OVERHEAT": "PV module or inverter thermal stress is above normal operating range",
        }
        return severity, earth_messages.get(label, space_message)

    def _update_header(self, metrics: dict, active_alerts: list[tuple[int, str, float]]) -> None:
        if active_alerts:
            alert_text = " | ".join(f"A{array_id} {label.replace('_', ' ')}" for array_id, label, _ in active_alerts[:3])
            self.status_label.configure(text=alert_text, fg=COLORS["red"])
            self.alert_label.configure(text=alert_text, fg=COLORS["red"])
        else:
            self.status_label.configure(text="Nominal", fg=COLORS["green"])
            self.alert_label.configure(text="No active alerts", fg=COLORS["green"])

        self.mode_label.configure(text=metrics["power_mode"])
        self.battery_label.configure(text=f"{metrics['battery_charge']:.1f}%")
        self.environment_label.configure(text="Earth" if metrics["environment_mode"] == "earth" else "Space")
        self.device_label.configure(text=cuda_summary())

    def _update_array_cards(self, readings) -> None:
        for reading in readings:
            labels = self.array_labels[reading.array_id]
            predicted, confidence = self.latest_predictions.get(reading.array_id, ("--", 0.0))
            prediction_color = COLORS["green"] if predicted == "NORMAL" else COLORS["yellow"]
            if predicted in IMPACT_CLASSES or reading.actual_label in IMPACT_CLASSES:
                prediction_color = COLORS["red"]
            labels["Power"].configure(text=f"{reading.power:,.0f} W")
            labels["Efficiency"].configure(text=f"{reading.efficiency:.1f}%")
            labels["Voltage"].configure(text=f"{reading.voltage:.1f} V")
            labels["Current"].configure(text=f"{reading.current:.2f} A")
            labels["Temp"].configure(text=f"{reading.temperature:.1f} C")
            labels["Exposure"].configure(text=f"{reading.exposure_factor:.0%}")
            labels["Vibration"].configure(text=f"{reading.vibration_g:.3f} g")
            labels["Prediction"].configure(text=f"{predicted.replace('_', ' ')}\n{confidence:.0%}", fg=prediction_color)

    def _iss_payload_power(self, metrics: dict) -> float:
        if metrics.get("environment_mode") == "earth":
            return 0.0
        fixed_charge = float(metrics.get("fixed_charge_power", 0.0))
        total = float(metrics.get("total_power", 0.0))
        base = float(metrics.get("base_load", 32_000.0))
        return min(STARLAB_PAYLOAD_POWER_W, max(0.0, total - fixed_charge - base))

    def _update_benchmark_accounting(self, metrics: dict, dt_seconds: float) -> None:
        if metrics.get("environment_mode") == "earth":
            return
        extra_payload = max(0.0, float(metrics.get("payload_power", 0.0)) - self._iss_payload_power(metrics))
        self.benchmark_extra_payload_wh += extra_payload * dt_seconds / 3600.0

    def _update_benchmark_tab(self, metrics: dict) -> None:
        if not self.benchmark_labels:
            return
        if metrics.get("environment_mode") == "earth":
            for key, label in self.benchmark_labels.items():
                label.configure(text="Earth mode uses separate PV benchmarks" if key == "mode" else "--")
            return

        iss_payload = self._iss_payload_power(metrics)
        our_payload = float(metrics.get("payload_power", 0.0))
        extra_payload = max(0.0, our_payload - iss_payload)
        event_count = len(self.benchmark_avoided_eva_events)
        eva_hours = event_count * 2.0
        low = eva_hours * 50_000
        mid = eva_hours * 75_000
        high = eva_hours * 100_000
        self.benchmark_labels["mode"].configure(text="Starlab dynamic routing vs ISS fixed 60% rule")
        self.benchmark_labels["iss_charge"].configure(text=f"{metrics['fixed_charge_power'] / 1000:.1f} kW")
        self.benchmark_labels["our_charge"].configure(text=f"{metrics['dynamic_charge_power'] / 1000:.1f} kW")
        self.benchmark_labels["iss_payload"].configure(text=f"{iss_payload / 1000:.1f} kW")
        self.benchmark_labels["our_payload"].configure(text=f"{our_payload / 1000:.1f} kW")
        self.benchmark_labels["extra_payload"].configure(text=f"+{extra_payload / 1000:.1f} kW")
        self.benchmark_labels["extra_energy"].configure(text=f"{self.benchmark_extra_payload_wh / 1000:.3f} kWh")
        self.benchmark_labels["avoided_events"].configure(text=str(event_count))
        self.benchmark_labels["eva_hours"].configure(text=f"{eva_hours:.1f} hr")
        self.benchmark_labels["eva_low"].configure(text=f"${low:,.0f}")
        self.benchmark_labels["eva_mid"].configure(text=f"${mid:,.0f}")
        self.benchmark_labels["eva_high"].configure(text=f"${high:,.0f}")
        self.benchmark_labels["latency"].configure(text=f"{metrics['local_latency_ms']:.2f} ms vs 300-800 ms Earth-link")
        self.benchmark_labels["summary"].configure(
            text=f"{self.benchmark_extra_payload_wh / 1000:.3f} kWh extra science energy + ${mid:,.0f} estimated EVA savings"
        )

    def _update_routing_views(self, metrics: dict, readings) -> None:
        if self.routing_labels:
            self.routing_labels["dynamic_charge_power"].configure(text=f"{metrics['dynamic_charge_power'] / 1000:.1f} kW")
            self.routing_labels["fixed_charge_power"].configure(text=f"{metrics['fixed_charge_power'] / 1000:.1f} kW")
            self.routing_labels["payload_power"].configure(text=f"{metrics['payload_power'] / 1000:.1f} kW")
            self.routing_labels["science_headroom"].configure(text=f"{metrics['science_headroom'] / 1000:.1f} kW")
            self.routing_labels["battery_priority_fraction"].configure(text=f"{metrics['battery_priority_fraction']:.0%}")
            self.routing_labels["local_latency_ms"].configure(text=f"{metrics['local_latency_ms']:.2f} ms")

        if self.mesh_tree is not None:
            for item in self.mesh_tree.get_children():
                self.mesh_tree.delete(item)
            for reading in readings:
                self.mesh_tree.insert(
                    "",
                    tk.END,
                    values=(
                        f"A{reading.array_id}",
                        f"{reading.power / 1000:.1f}",
                        f"{reading.benchmark_ratio:.2f}",
                        f"{reading.vibration_g:.3f}",
                        f"{reading.exposure_factor:.0%}",
                    ),
                )

        if "mesh_summary" in self.mission_labels:
            if readings:
                max_vibration = max(row.vibration_g for row in readings)
                low_ratio = min(row.benchmark_ratio for row in readings)
            else:
                max_vibration = 0.0
                low_ratio = 1.0
            mode_text = "on-station sensor mesh" if metrics["environment_mode"] == "space" else "ground PV sensor mesh"
            self.mission_labels["mesh_summary"].configure(
                text=(
                    f"{mode_text} streaming locally.\n"
                    f"Highest vibration: {max_vibration:.3f} g\n"
                    f"Lowest benchmark ratio: {low_ratio:.2f}\n"
                    f"Earth-link latency avoided: 300-800 ms"
                )
            )

    def _draw_network(self, metrics: dict, active_alerts: list[tuple[int, str, float]]) -> None:
        canvas = self.network_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 640)
        height = max(canvas.winfo_height(), 360)
        earth_mode = metrics["environment_mode"] == "earth"
        title = "Terrestrial Microgrid Routing" if earth_mode else "Orbital Power Routing"
        canvas.create_text(18, 18, anchor="nw", text=title, fill=COLORS["text"], font=("Segoe UI", 15, "bold"))
        if earth_mode:
            weather_text = f"Cloud {metrics['cloud_cover']:.0%}  Humidity {metrics['humidity']:.0%}"
            canvas.create_text(18, 44, anchor="nw", text=weather_text, fill=COLORS["muted"], font=("Segoe UI", 9))

        positions = {
            1: (70, 85),
            2: (width - 190, 85),
            3: (70, height - 130),
            4: (width - 190, height - 130),
        }
        battery = (width // 2 - 105, height // 2 - 58, width // 2 + 105, height // 2 + 58)
        load = (width // 2 - 75, height - 72, width // 2 + 75, height - 28)

        for reading in self.latest_readings:
            x, y = positions[reading.array_id]
            color = ARRAY_COLORS[reading.array_id - 1]
            if reading.actual_label in SEVERITY:
                color = COLORS["red"]
            canvas.create_rectangle(x, y, x + 120, y + 62, fill=COLORS["panel_alt"], outline=color, width=3)
            unit_name = "PV STRING" if earth_mode else "ARRAY"
            canvas.create_text(x + 60, y + 18, text=f"{unit_name} {reading.array_id}", fill=color, font=("Segoe UI", 9, "bold"))
            canvas.create_text(x + 60, y + 42, text=f"{reading.power / 1000:.1f} kW", fill=COLORS["text"], font=("Segoe UI", 11))
            canvas.create_line(x + 60, y + 31, width // 2, height // 2, fill=color, width=2, arrow=tk.LAST)

        mode_color = COLORS["green"] if metrics["power_mode"] in {"BATTERY PRIORITY", "BATTERY CHARGE"} else COLORS["yellow"]
        if metrics["power_mode"] in {"SHADOW RESERVE", "NIGHT RESERVE"}:
            mode_color = COLORS["blue"]
        canvas.create_rectangle(*battery, fill=COLORS["bg"], outline=mode_color, width=3)
        battery_name = "GROUND BATTERY BANK" if earth_mode else "SOLID STATE BATTERY"
        canvas.create_text(width // 2, battery[1] + 20, text=battery_name, fill=COLORS["text"], font=("Segoe UI", 9, "bold"))
        canvas.create_text(width // 2, battery[1] + 52, text=f"{metrics['battery_charge']:.1f}%", fill=mode_color, font=("Segoe UI", 24, "bold"))
        canvas.create_text(
            width // 2,
            battery[3] - 15,
            text=f"Charge {metrics['dynamic_charge_power'] / 1000:.1f} kW",
            fill=COLORS["muted"],
            font=("Segoe UI", 9),
        )

        canvas.create_rectangle(*load, fill=COLORS["panel_alt"], outline=COLORS["text"])
        load_name = "SITE LOAD" if earth_mode else "STATION LOAD"
        canvas.create_text((load[0] + load[2]) // 2, (load[1] + load[3]) // 2, text=load_name, fill=COLORS["text"], font=("Segoe UI", 10, "bold"))
        canvas.create_line(width // 2, battery[3], width // 2, load[1], fill=mode_color, width=3, arrow=tk.LAST)
        if not earth_mode:
            canvas.create_text(
                width // 2,
                load[3] + 18,
                text=f"Payload {metrics['payload_power'] / 1000:.1f} kW / {metrics['payload_target'] / 1000:.0f} kW",
                fill=COLORS["muted"],
                font=("Segoe UI", 9),
            )
            canvas.create_text(
                width // 2,
                load[3] + 36,
                text=f"ISS fixed rule would send {metrics['fixed_charge_power'] / 1000:.1f} kW to batteries",
                fill=COLORS["muted"],
                font=("Segoe UI", 8),
            )

        orbit = "DAYLIGHT" if earth_mode and metrics["sunlight"] else "NIGHT" if earth_mode else "SUNLIGHT" if metrics["sunlight"] else "SHADOW"
        orbit_color = COLORS["yellow"] if metrics["sunlight"] else COLORS["blue"]
        canvas.create_oval(width - 96, 20, width - 32, 84, fill=orbit_color, outline="")
        canvas.create_text(width - 64, 52, text=orbit, fill=COLORS["bg"], font=("Segoe UI", 8, "bold"), width=58)
        timer = metrics["time_to_orbit_change"]
        timer_text = "Manual" if timer == 0 else f"{timer // 60:02d}:{timer % 60:02d}"
        canvas.create_text(width - 64, 100, text=timer_text, fill=COLORS["text"], font=("Segoe UI", 10))

        if active_alerts:
            canvas.create_text(
                width // 2,
                52,
                text="ACTIVE ALERT",
                fill=COLORS["red"],
                font=("Segoe UI", 15, "bold"),
            )

    def _draw_charts(self) -> None:
        canvas = self.chart_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 640)
        height = max(canvas.winfo_height(), 220)
        canvas.create_text(18, 14, anchor="nw", text="System Trend", fill=COLORS["text"], font=("Segoe UI", 14, "bold"))
        mid = 58 + (height - 92) // 2
        self._draw_series(canvas, self.power_history, 70, mid - 8, width - 30, 58, COLORS["cyan"], "Power W")
        self._draw_series(canvas, self.efficiency_history, 70, height - 42, width - 30, mid + 24, COLORS["green"], "Efficiency %")

    def _draw_series(self, canvas: tk.Canvas, values: list[float], x1: int, y1: int, x2: int, y2: int, color: str, label: str) -> None:
        canvas.create_line(x1, y1, x2, y1, fill=COLORS["line"])
        canvas.create_text(x1, y2 - 18, anchor="w", text=label, fill=color, font=("Segoe UI", 9, "bold"))
        if len(values) < 2:
            return
        minimum = min(values)
        maximum = max(values)
        span = max(1.0, maximum - minimum)
        points = []
        for index, value in enumerate(values):
            x = x1 + (x2 - x1) * index / max(1, len(values) - 1)
            y = y1 - (y1 - y2) * (value - minimum) / span
            points.extend([x, y])
        canvas.create_line(*points, fill=color, width=2, smooth=True)
        canvas.create_text(x2, y2 - 18, anchor="e", text=f"{values[-1]:.1f}", fill=color, font=("Segoe UI", 10, "bold"))

    def _refresh_database_views(self) -> None:
        counts = database_counts()
        self.db_summary.configure(
            text=(
                f"sensor_readings : {counts['sensor_readings']}\n"
                f"incidents       : {counts['incidents']}\n"
                f"system_metrics  : {counts['system_metrics']}"
            )
        )

        self.reading_list.delete(0, tk.END)
        for row in recent_readings(12):
            mode = "E" if row["environment_mode"] == "earth" else "S"
            self.reading_list.insert(
                tk.END,
                f"{row['timestamp'][-9:-1]} {mode} A{row['array_id']} {row['power'] / 1000:5.1f}kW "
                f"{row['efficiency']:5.1f}% {row['predicted_label']}",
            )

        for item in self.incident_tree.get_children():
            self.incident_tree.delete(item)
        for row in recent_incidents(24):
            self.incident_tree.insert(
                "",
                tk.END,
                values=(row["timestamp"][11:19], f"A{row['array_id']}", row["incident_type"], row["severity"]),
            )


def main() -> None:
    root = tk.Tk()
    StarlabDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
