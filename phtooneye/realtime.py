import csv
import os
import shutil
import smtplib
import ssl
import time
from collections import deque
from datetime import datetime
from email.message import EmailMessage

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pytz
import serial

# port and output folder
output_dir = "/Users/marcus/Desktop/Arduino"
arduino_port = "/dev/cu.usbmodem2301"  # use the board COM port on Windows
baud_rate = 115200

# settings for the live plot
log_prefix = "intensity_log"
time_zone = "UTC"
plot_window = 60  # number of points shown
rolling_window = 50  # points used for anomaly detection
bin_size = 100  # one sample per ms, so about 0.1 s per point
max_bin_seconds = 0.5  # seconds to wait for one bin
update_interval_ms = 100  # plot refresh interval
auto_y = False  # resize the y-axis as readings change
y_min = 0
y_max = 1000
y_tick_step = 100
x_tick_target = 4

# trend and forecast
trend_window = 12
trend_slope_eps = 0.5
forecast_steps = 6

# health score weights
health_window = 20
health_weight_sample = 0.6
health_weight_anomaly = 0.25
health_weight_drop = 0.15

# moving average and alert markers
moving_avg_window = 10
alert_rolling_window = 20
show_event_lines = True
event_line_alpha = 0.15

# folders and snapshots
snapshot_every_n = 50  # set to 0 to disable
save_final_snapshot = False
snapshot_dir_name = "snapshots"
csv_dir_name = "csv_logs"
summary_dir_name = "summaries"
bundle_dir_name = "bundles"

# expected intensity range
show_target_band = False
target_min = 200
target_max = 800

# calibration
cal_slope = 1.0
cal_offset = 0.0

# email settings come from email.env or the environment
email_to = ""
email_subject_prefix = "Astronaut Run Bundle"
smtp_host = ""
smtp_port = 587
smtp_user = ""
smtp_pass = ""
smtp_use_tls = True
email_from = ""
email_enabled = False
auto_delete_bundle_after_email = True
email_config_path = os.environ.get("ASTRO_EMAIL_CONFIG", "")

# optional second reading
aux_label = "Aux"
aux_color = "#6c5ce7"
aux_drop_abs = 5.0
aux_rise_abs = 5.0

# limits for sudden jumps
drop_abs = 50.0
drop_pct = 0.2
drop_cooldown_s = 3.0
rise_abs = 50.0
rise_pct = 0.2
rise_cooldown_s = 3.0
alert_display_seconds = 6.0
alert_line_max = 3

# filtering and smoothing
filter_method = "mad"  # "mad" or "zscore"
outlier_z = 2.5
mad_threshold = 3.5
trim_ratio = 0.1
ema_alpha = 0.3
use_ema_for_display = True
show_raw_trace = False
raw_trace_alpha = 0.25
anomaly_use_smoothed = True
trend_use_smoothed = True
drop_use_smoothed = True
rise_use_smoothed = True

# check for outliers against earlier points
anomaly_limit = 3.0
anomaly_min_history = 20
alert_cooldown_s = 2.0

ser = serial.Serial(arduino_port, baud_rate, timeout=1)
ser.reset_input_buffer()

# use UTC for timestamps in the csv
mission_start_utc = datetime.now(pytz.utc)
zone = pytz.timezone(time_zone)
run_id = mission_start_utc.strftime("%Y%m%d_%H%M%S")

os.makedirs(output_dir, exist_ok=True)
snapshot_dir = os.path.join(output_dir, snapshot_dir_name)
os.makedirs(snapshot_dir, exist_ok=True)
csv_dir = os.path.join(output_dir, csv_dir_name)
os.makedirs(csv_dir, exist_ok=True)
summary_dir = os.path.join(output_dir, summary_dir_name)
os.makedirs(summary_dir, exist_ok=True)
bundle_dir = os.path.join(output_dir, bundle_dir_name)
os.makedirs(bundle_dir, exist_ok=True)
log_path = os.path.join(csv_dir, f"{log_prefix}_{run_id}.csv")
log_file = open(log_path, "w", newline="")
log_writer = csv.writer(log_file)
csv_columns = ["timestamp_utc", "mission_elapsed", "avg_intensity", "ema_intensity", "avg_aux",
               "raw_mean", "raw_std", "sample_count", "filtered_count", "outlier_count",
               "sample_ratio", "quality", "anomaly_z", "trend", "trend_slope",
               "drop_flag", "drop_reason", "spike_flag", "spike_reason", "health_score"]
log_writer.writerow(csv_columns)
log_counter = 0

# keep the most recent points for the plot
time_x = deque(maxlen=plot_window)
intensity_y = deque(maxlen=plot_window)
intensity_smooth_y = deque(maxlen=plot_window)
aux_y = deque(maxlen=plot_window)
anomaly_flags = deque(maxlen=plot_window)
drop_flags = deque(maxlen=plot_window)
spike_flags = deque(maxlen=plot_window)
rolling_values = deque(maxlen=rolling_window)
last_alert_time = 0.0
last_drop_time = 0.0
last_valid_for_drop = None
last_rise_time = 0.0
last_valid_for_rise = None
ema_intensity = None
alert_messages = deque()
health_sample_window = deque(maxlen=health_window)
health_anomaly_window = deque(maxlen=health_window)
health_drop_window = deque(maxlen=health_window)

# save the points for the end-of-run summary
run_intensity = []
run_intensity_smooth = []
run_aux = []
run_sample_ratio = []
run_health = []
run_anomaly = []
run_drop = []
run_spike = []
run_snapshot_files = []

# biggest drop and peak seen so far
max_drop_magnitude = None
max_drop_time = None
max_spike_magnitude = None
max_spike_time = None
quality_counts = {"GOOD": 0, "PARTIAL": 0, "NO DATA": 0, "ALL OUTLIERS": 0}
trend_counts = {"RISING": 0, "FALLING": 0, "STEADY": 0, "N/A": 0}

fig = plt.figure(figsize=(14, 7), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax_aux = ax.twinx()

def remove_z_outliers(data, threshold=2):
    if not data:
        return data
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data

    return [value for value in data if abs((value - mean) / std) <= threshold]


def remove_mad_outliers(data, threshold=3.5):
    if not data:
        return data
    median = np.median(data)
    abs_dev = [abs(x - median) for x in data]
    mad = np.median(abs_dev)
    if mad == 0:
        return data
    return [
        value
        for value in data
        if abs(0.6745 * (value - median) / mad) <= threshold
    ]


def filter_outliers(data):
    if filter_method == "mad":
        return remove_mad_outliers(data, mad_threshold)
    return remove_z_outliers(data, outlier_z)


def trimmed_mean(data, trim_ratio):
    if not data:
        return float("nan")
    if trim_ratio <= 0:
        return float(np.mean(data))
    sorted_vals = sorted(data)
    trim_n = int(len(sorted_vals) * trim_ratio)
    if trim_n == 0 or trim_n * 2 >= len(sorted_vals):
        return float(np.mean(sorted_vals))
    return float(np.mean(sorted_vals[trim_n:-trim_n]))


def format_met(delta):
    total = int(delta.total_seconds())
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def is_anomaly(value, history, z_threshold, min_history):
    if len(history) < min_history:
        return False, 0.0
    mean = np.mean(history)
    std = np.std(history)
    if std == 0:
        return False, 0.0
    z = (value - mean) / std
    return abs(z) >= z_threshold, z


def compute_health_score(sample_ratios, anomaly_flags, drop_flags):
    if len(sample_ratios) == 0:
        return float("nan")
    sample_avg = float(np.mean(sample_ratios))
    anomaly_rate = float(np.mean(anomaly_flags)) if len(anomaly_flags) else 0.0
    drop_rate = float(np.mean(drop_flags)) if len(drop_flags) else 0.0
    sample_score = health_weight_sample * sample_avg
    anomaly_score = health_weight_anomaly * (1 - anomaly_rate)
    drop_score = health_weight_drop * (1 - drop_rate)
    score = 100 * (sample_score + anomaly_score + drop_score)
    return max(0.0, min(100.0, score))


def moving_average(series, window):
    if window <= 1:
        return [float(x) if not np.isnan(x) else float("nan") for x in series]
    out = [float("nan")] * len(series)
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_vals = [v for v in series[start:i + 1] if not np.isnan(v)]
        if len(window_vals) == 0:
            continue
        out[i] = float(np.mean(window_vals))
    return out


def rolling_rate(flags, window):
    if window <= 1:
        return [float(f) for f in flags]
    rates = []
    for i in range(len(flags)):
        start = max(0, i - window + 1)
        window_flags = flags[start:i + 1]
        rate = sum(1 for f in window_flags if f) / max(1, len(window_flags))
        rates.append(rate)
    return rates


def load_env_file(path):
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


def refresh_email_config(default_env_path):
    global email_to, smtp_host, smtp_port, smtp_user, smtp_pass, email_from, email_enabled
    config_path = email_config_path or default_env_path
    load_env_file(config_path)
    email_to = os.environ.get("ASTRO_EMAIL_TO", "")
    smtp_host = os.environ.get("ASTRO_SMTP_HOST", "")
    smtp_port = int(os.environ.get("ASTRO_SMTP_PORT", "587"))
    smtp_user = os.environ.get("ASTRO_SMTP_USER", "")
    smtp_pass = os.environ.get("ASTRO_SMTP_PASS", "")
    email_from = os.environ.get("ASTRO_EMAIL_FROM", smtp_user if smtp_user else email_to)
    email_enabled = email_to != "" and smtp_host != ""
    return config_path


def send_email_with_attachment(to_addr, subject, body, attachment_path):
    if not smtp_host or not to_addr or not attachment_path:
        return False
    msg = EmailMessage()
    msg["From"] = email_from
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    with open(attachment_path, "rb") as file_obj:
        data = file_obj.read()
        msg.add_attachment(data, maintype="application", subtype="zip", filename=os.path.basename(attachment_path))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
        if smtp_use_tls:
            server.starttls(context=ssl.create_default_context())
        if smtp_user:
            server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    return True


def read_bin():
    readings = []
    aux_readings = []

    # Keep whatever arrived before the deadline so a slow board can still be plotted.
    started = time.monotonic()
    while len(readings) < bin_size and time.monotonic() - started < max_bin_seconds:
        if not ser.in_waiting:
            time.sleep(0.001)
            continue

        line = ser.readline().decode("utf-8").strip()
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if not parts:
            continue

        try:
            intensity = float(parts[0])
        except ValueError:
            return None  # A bad primary reading invalidates this bin.
        readings.append(intensity * cal_slope + cal_offset)

        if len(parts) > 1:
            try:
                aux_readings.append(float(parts[1]))
            except ValueError:
                pass  # the second reading is optional

    return readings, aux_readings


def animate(_):
    global last_alert_time, log_counter, alert_messages
    global last_drop_time, last_valid_for_drop, ema_intensity
    global last_rise_time, last_valid_for_rise
    global max_drop_magnitude, max_drop_time, max_spike_magnitude, max_spike_time

    now_utc = datetime.now(pytz.utc)
    local_time = now_utc.astimezone(zone)
    formatted_time = local_time.strftime("%H:%M:%S.%f")[:-3]
    met = now_utc - mission_start_utc
    met_str = format_met(met)

    batch = read_bin()
    if batch is None:
        return
    readings, aux_readings = batch

    sample_count = len(readings)
    filtered_data = filter_outliers(readings)
    outlier_count = sample_count - len(filtered_data)
    raw_mean = float(np.mean(readings)) if readings else float("nan")
    raw_std = float(np.std(readings)) if readings else float("nan")

    quality = "GOOD"
    if not readings:
        quality = "NO DATA"
    elif not filtered_data:
        quality = "ALL OUTLIERS"
    elif sample_count < bin_size:
        quality = "PARTIAL"

    avg_intensity = trimmed_mean(filtered_data, trim_ratio)
    avg_aux = trimmed_mean(filter_outliers(aux_readings), trim_ratio)

    # Carry the last smoothed value through gaps in the serial stream.
    if not np.isnan(avg_intensity):
        if ema_intensity is None:
            ema_intensity = avg_intensity
        else:
            ema_intensity = ema_alpha * avg_intensity + (1 - ema_alpha) * ema_intensity
    ema_value = ema_intensity if ema_intensity is not None else float("nan")

    time_x.append(formatted_time)
    intensity_y.append(avg_intensity)
    intensity_smooth_y.append(ema_value)
    aux_y.append(avg_aux)

    # look for sudden drops and rises
    drop_flag = False
    drop_reason = ""
    spike_flag = False
    spike_reason = ""
    clock = time.monotonic()
    value_for_drop = ema_value if drop_use_smoothed else avg_intensity
    if not np.isnan(value_for_drop):
        if last_valid_for_drop is not None:
            delta = value_for_drop - last_valid_for_drop
            pct = delta / max(abs(last_valid_for_drop), 1.0)
            if (delta <= -drop_abs and pct <= -drop_pct and (clock - last_drop_time) > drop_cooldown_s):
                drop_flag = True
                last_drop_time = clock
                aux_drop = False
                if (not np.isnan(avg_aux) and len(aux_y) >= 2 and not np.isnan(aux_y[-2]) and (avg_aux - aux_y[-2]) <= -aux_drop_abs):
                    aux_drop = True
                if aux_drop:
                    drop_reason = "Possible lighting decrease or occlusion (aux dropped)."
                else:
                    drop_reason = "Possible occlusion, angle change, lighting change, or sensor interruption."
                if max_drop_magnitude is None or abs(delta) > max_drop_magnitude:
                    max_drop_magnitude = abs(delta)
                    max_drop_time = now_utc
        last_valid_for_drop = value_for_drop

    value_for_spike = ema_value if rise_use_smoothed else avg_intensity
    if not np.isnan(value_for_spike):
        if last_valid_for_rise is not None:
            delta = value_for_spike - last_valid_for_rise
            pct = delta / max(abs(last_valid_for_rise), 1.0)
            if (delta >= rise_abs and pct >= rise_pct and (clock - last_rise_time) > rise_cooldown_s):
                spike_flag = True
                last_rise_time = clock
                aux_rise = False
                if (not np.isnan(avg_aux) and len(aux_y) >= 2 and not np.isnan(aux_y[-2]) and (avg_aux - aux_y[-2]) >= aux_rise_abs):
                    aux_rise = True
                if aux_rise:
                    spike_reason = "Possible direct lighting increase (aux rose)."
                else:
                    spike_reason = "Possible sudden lighting, reflection, or sensor saturation."
                if max_spike_magnitude is None or abs(delta) > max_spike_magnitude:
                    max_spike_magnitude = abs(delta)
                    max_spike_time = now_utc
        last_valid_for_rise = value_for_spike

    # Check against earlier points before adding this one to the baseline.
    anomaly_flag = False
    anomaly_z = 0.0
    value_for_anomaly = ema_value if anomaly_use_smoothed else avg_intensity
    if not np.isnan(value_for_anomaly):
        anomaly_flag, anomaly_z = is_anomaly(value_for_anomaly, rolling_values, anomaly_limit, anomaly_min_history)
        rolling_values.append(value_for_anomaly)

    anomaly_flags.append(anomaly_flag)
    drop_flags.append(drop_flag)
    spike_flags.append(spike_flag)

    # Count peaks and drops together for the health score.
    sample_ratio = sample_count / bin_size if bin_size else 0.0
    health_sample_window.append(sample_ratio)
    health_anomaly_window.append(1.0 if anomaly_flag else 0.0)
    health_drop_window.append(1.0 if (drop_flag or spike_flag) else 0.0)
    health_score = compute_health_score(
        health_sample_window, health_anomaly_window, health_drop_window
    )

    ax.clear()
    ax_aux.clear()
    ax_aux.patch.set_alpha(0)

    x_labels = list(time_x)
    intensity_raw_list = list(intensity_y)
    intensity_smooth_list = list(intensity_smooth_y)
    aux_list = list(aux_y)
    x_positions = list(range(len(x_labels)))
    plot_values = intensity_smooth_list if use_ema_for_display else intensity_raw_list
    plot_label = "Intensity (EMA)" if use_ema_for_display else "Intensity"
    ma_values = moving_average(plot_values, moving_avg_window)

    if show_target_band:
        ax.axhspan(target_min, target_max, color="#e3f2fd", alpha=0.35, label="Target Band")

    # Skip missing values but keep their positions so the slope is per bin.
    trend_label = "STEADY"
    trend_slope = 0.0
    trend_color = "#fff8e1"
    has_trend = False

    trend_values = intensity_smooth_list if trend_use_smoothed else intensity_raw_list
    n_points = len(plot_values)
    start_idx = max(0, n_points - trend_window)
    x_window = []
    y_window = []
    for idx in range(start_idx, n_points):
        y_val = trend_values[idx]
        if not np.isnan(y_val):
            x_window.append(idx)
            y_window.append(y_val)

    if len(y_window) >= 2:
        slope, intercept = np.polyfit(x_window, y_window, 1)
        trend_slope = float(slope)
        has_trend = True
        if slope > trend_slope_eps:
            trend_label = "RISING"
            trend_color = "#e6f4ea"
        elif slope < -trend_slope_eps:
            trend_label = "FALLING"
            trend_color = "#fdecea"
        else:
            trend_label = "STEADY"
            trend_color = "#fff8e1"

        trend_x = np.arange(start_idx, n_points)
        trend_y = slope * trend_x + intercept
        ax.plot(trend_x, trend_y, color="#666666", linestyle="--", linewidth=1)

        if forecast_steps > 0:
            forecast_x = np.arange(n_points - 1, n_points - 1 + forecast_steps + 1)
            forecast_y = slope * forecast_x + intercept
            ax.plot(forecast_x, forecast_y, color="#999999", linestyle=":", linewidth=1.5)

    ax.plot(x_positions, plot_values, color="#1b6ca8", linewidth=2, label=plot_label)
    ax.plot(x_positions, ma_values, color="#0b7285", linewidth=1.5, linestyle="--", label=f"MA({moving_avg_window})")
    if show_raw_trace and use_ema_for_display:
        ax.plot(x_positions, intensity_raw_list, color="#1b6ca8", linewidth=1, alpha=raw_trace_alpha, label="Raw")

    if show_event_lines:
        for flags, color in [(anomaly_flags, "#b30000"), (drop_flags, "#d97706"), (spike_flags, "#2f9e44")]:
            for i, flagged in enumerate(flags):
                if flagged:
                    ax.axvline(i, color=color, alpha=event_line_alpha, linewidth=1)

    low, high = y_min, y_max
    if auto_y:
        y_values = [v for v in intensity_y if not np.isnan(v)]
        if len(y_values) > 1:
            low, high = min(y_values), max(y_values)
            pad = max(1.0, (high - low) * 0.2)
            low, high = low - pad, high + pad
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_step))

    if anomaly_flag and (clock - last_alert_time) > alert_cooldown_s:
        last_alert_time = clock
        alert_messages.append((clock + alert_display_seconds, f"ANOMALY z={anomaly_z:.2f}"))
    if drop_flag:
        alert_messages.append((clock + alert_display_seconds, f"SUDDEN DROP: {drop_reason}"))
    if spike_flag:
        alert_messages.append((clock + alert_display_seconds, f"SUDDEN PEAK: {spike_reason}"))

    alert_messages = deque([(exp, msg) for exp, msg in alert_messages if exp > clock])

    if drop_flag:
        ax.set_facecolor("#fff4e5")
    elif spike_flag:
        ax.set_facecolor("#e6ffed")
    else:
        ax.set_facecolor(trend_color)
    border, width = "#333333", 1
    if anomaly_flag:
        border, width = "#b30000", 2
    elif drop_flag:
        border, width = "#d97706", 2
    elif spike_flag:
        border, width = "#2f9e44", 2
    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(width)

    ax.set_xlabel(f"Time ({time_zone})")
    ax.set_ylabel("Intensity (Brightness)")
    ax.set_title("Intensity vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    if any(not np.isnan(value) for value in aux_list):
        ax_aux.plot(x_positions, aux_list, color=aux_color, linewidth=1.5, alpha=0.8)
        ax_aux.set_ylabel(aux_label, color=aux_color)
        ax_aux.tick_params(axis="y", labelcolor=aux_color)
        ax_aux.relim()
        ax_aux.autoscale_view()
    else:
        ax_aux.set_yticks([])
        ax_aux.set_ylabel("")

    x_max = max(1, n_points - 1 + (forecast_steps if has_trend else 0))
    ax.set_xlim(0, x_max)
    ax_aux.set_xlim(ax.get_xlim())

    status_text = f"UTC: {now_utc.strftime('%H:%M:%S')}  MET: {met_str}"
    status_text += f"  Quality: {quality}"
    if has_trend:
        status_text += f"  Trend: {trend_label} ({trend_slope:.2f}/pt)"
    else:
        status_text += "  Trend: N/A"
    active_alerts = [msg for _, msg in list(alert_messages)[-alert_line_max:]]
    if active_alerts:
        status_text += "\nALERTS: " + " | ".join(active_alerts)

    status_text += f"\nHealth: {health_score:.1f}"

    status_box = {"facecolor": "white", "alpha": 0.7, "edgecolor": "none"}
    ax.text(0.02, 0.95, status_text, transform=ax.transAxes, fontsize=9, va="top", bbox=status_box)

    if drop_flag and n_points > 0 and not np.isnan(plot_values[-1]):
        ax.scatter([n_points - 1], [plot_values[-1]], color="#d97706", s=60, zorder=5, marker="v")
    if spike_flag and n_points > 0 and not np.isnan(plot_values[-1]):
        ax.scatter([n_points - 1], [plot_values[-1]], color="#2f9e44", s=60, zorder=5, marker="^")

    if len(x_labels) > 0:
        step = max(1, len(x_labels) // x_tick_target)
        tick_positions = list(range(0, len(x_labels), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([x_labels[i] for i in tick_positions], rotation=45, ha="right")

    quality_counts[quality] += 1
    trend = trend_label if has_trend else "N/A"
    trend_counts[trend] += 1

    run_intensity.append(avg_intensity)
    run_intensity_smooth.append(ema_value)
    run_aux.append(avg_aux)
    run_sample_ratio.append(sample_ratio)
    run_health.append(health_score)
    run_anomaly.append(anomaly_flag)
    run_drop.append(drop_flag)
    run_spike.append(spike_flag)

    # Leave missing measurements blank so the CSV can be imported as numeric data.
    row = [now_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), met_str]
    for value in [avg_intensity, ema_value, avg_aux, raw_mean, raw_std]:
        row.append("" if np.isnan(value) else f"{value:.6f}")

    z_text = f"{anomaly_z:.2f}" if anomaly_flag else ""
    slope_text = f"{trend_slope:.2f}" if has_trend else ""
    health_text = "" if np.isnan(health_score) else f"{health_score:.2f}"
    row += [sample_count, len(filtered_data), outlier_count, f"{sample_ratio:.2f}", quality]
    row += [z_text, trend, slope_text, int(drop_flag), drop_reason, int(spike_flag), spike_reason]
    row.append(health_text)
    log_writer.writerow(row)
    log_counter += 1
    if log_counter % 25 == 0:
        log_file.flush()

    if snapshot_every_n and log_counter % snapshot_every_n == 0:
        snapshot_path = os.path.join(snapshot_dir, f"{log_prefix}_{run_id}_snap_{log_counter:06d}.png")
        fig.savefig(snapshot_path, dpi=150)
        run_snapshot_files.append(snapshot_path)


def save_run(end_time_utc):
    duration = end_time_utc - mission_start_utc
    summary_plot_path = None
    bundle_root = None
    summary_path = os.path.join(summary_dir, f"{log_prefix}_{run_id}_summary.txt")
    total_points = len(run_intensity)
    anomaly_count = sum(bool(flag) for flag in run_anomaly)
    drop_count = sum(run_drop)
    spike_count = sum(run_spike)
    alert_count = anomaly_count + drop_count + spike_count

    try:
        valid_points = sum(not np.isnan(value) for value in run_intensity)
        count = max(1, total_points)
        avg_sample_ratio = sum(run_sample_ratio) / count
        anomaly_rate = anomaly_count / count
        drop_rate = drop_count / count
        spike_rate = spike_count / count
        alert_rate = alert_count / count
        if avg_sample_ratio >= 0.95 and alert_rate <= 0.05:
            efficiency_grade = "NOMINAL"
        elif avg_sample_ratio >= 0.85 and alert_rate <= 0.10:
            efficiency_grade = "WATCH"
        else:
            efficiency_grade = "DEGRADED"
        with open(summary_path, "w") as summary_file:
            summary_file.write("Run Summary\n")
            summary_file.write("=" * 40 + "\n")
            summary_file.write(f"Start (UTC): {mission_start_utc.isoformat()}\n")
            summary_file.write(f"End   (UTC): {end_time_utc.isoformat()}\n")
            summary_file.write(f"Duration: {format_met(duration)}\n")
            summary_file.write(f"Calibration slope: {cal_slope}\n")
            summary_file.write(f"Calibration offset: {cal_offset}\n")
            summary_file.write(f"Total points: {total_points}\n")
            summary_file.write(f"Valid intensity points: {valid_points}\n")
            summary_file.write(f"Average sample ratio: {avg_sample_ratio:.2f}\n")
            summary_file.write(f"Anomaly rate: {anomaly_rate:.2f}\n")
            summary_file.write(f"Drop rate: {drop_rate:.2f}\n")
            summary_file.write(f"Spike rate: {spike_rate:.2f}\n")
            summary_file.write(f"Alert rate: {alert_rate:.2f}\n")
            summary_file.write(f"Efficiency grade: {efficiency_grade}\n")
            series_to_report = [("Intensity Stats", run_intensity),
                                ("Smoothed Intensity Stats", run_intensity_smooth), ("Aux Stats", run_aux)]
            for title, series in series_to_report:
                summary_file.write(f"\n{title}\n" + "-" * 40 + "\n")
                values = [value for value in series if not np.isnan(value)]
                if values:
                    summary_file.write(f"Min: {min(values)}\nMax: {max(values)}\n")
                    summary_file.write(f"Mean: {sum(values) / len(values)}\n")
                else:
                    summary_file.write("Min: n/a\nMax: n/a\nMean: n/a\n")

            summary_file.write("\nQuality Counts\n")
            summary_file.write("-" * 40 + "\n")
            for key in ["GOOD", "PARTIAL", "NO DATA", "ALL OUTLIERS"]:
                summary_file.write(f"{key}: {quality_counts.get(key, 0)}\n")
            summary_file.write("\nTrend Counts\n")
            summary_file.write("-" * 40 + "\n")
            for key in ["RISING", "FALLING", "STEADY", "N/A"]:
                summary_file.write(f"{key}: {trend_counts.get(key, 0)}\n")
            summary_file.write("\nAlerts\n")
            summary_file.write("-" * 40 + "\n")
            summary_file.write(f"Anomaly count: {anomaly_count}\n")
            summary_file.write(f"Sudden drop count: {drop_count}\n")
            summary_file.write(f"Sudden peak count: {spike_count}\n")
            summary_file.write(f"Alerts present: {'YES' if alert_count else 'NO'}\n")
            if max_drop_magnitude is not None and max_drop_time is not None:
                summary_file.write(f"Largest drop: {max_drop_magnitude:.2f} at {max_drop_time.isoformat()}\n")
            if max_spike_magnitude is not None and max_spike_time is not None:
                summary_file.write(f"Largest peak: {max_spike_magnitude:.2f} at {max_spike_time.isoformat()}\n")
            summary_file.write("\nHealth Score\n" + "-" * 40 + "\n")
            values = [value for value in run_health if not np.isnan(value)]
            if values:
                summary_file.write(f"Min: {min(values)}\nMax: {max(values)}\n")
                summary_file.write(f"Mean: {sum(values) / len(values)}\n")
            else:
                summary_file.write("Min: n/a\nMax: n/a\nMean: n/a\n")
    except Exception:
        pass

    if total_points > 0:
        try:
            fig_summary, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=150)
            x_vals = list(range(len(run_health)))

            ax_top = axes[0]
            ax_top.plot(x_vals, run_health, color="#2b8a3e", label="Health Score")
            ax_top.set_ylim(0, 100)
            ax_top.set_ylabel("Health Score")
            ax_top.grid(True, alpha=0.3)
            ax_top.legend(loc="upper left")

            ax_top_r = ax_top.twinx()
            ax_top_r.plot(x_vals, run_sample_ratio, color="#1b6ca8", alpha=0.7, label="Sample Ratio")
            ax_top_r.set_ylim(0, 1)
            ax_top_r.set_ylabel("Sample Ratio")
            ax_top_r.legend(loc="upper right")

            ax_mid = axes[1]
            anomaly_rate_series = rolling_rate(run_anomaly, alert_rolling_window)
            drop_rate_series = rolling_rate(run_drop, alert_rolling_window)
            spike_rate_series = rolling_rate(run_spike, alert_rolling_window)
            alert_rates = zip(anomaly_rate_series, drop_rate_series, spike_rate_series)
            combined_rate = [min(1.0, a + d + s) for a, d, s in alert_rates]
            ax_mid.plot(x_vals, anomaly_rate_series, color="#b30000", label="Anomaly Rate")
            ax_mid.plot(x_vals, drop_rate_series, color="#d97706", label="Drop Rate")
            ax_mid.plot(x_vals, spike_rate_series, color="#2f9e44", label="Peak Rate")
            ax_mid.plot(x_vals, combined_rate, color="#6c757d", linestyle="--", label="Combined Alert Rate")
            ax_mid.set_ylim(0, 1)
            ax_mid.set_ylabel("Alert Rate")
            ax_mid.grid(True, alpha=0.3)
            ax_mid.legend(loc="upper left")

            ax_mid_r = ax_mid.twinx()
            cum_anom = np.cumsum([1 if f else 0 for f in run_anomaly])
            cum_drop = np.cumsum([1 if f else 0 for f in run_drop])
            cum_spike = np.cumsum([1 if f else 0 for f in run_spike])
            ax_mid_r.plot(x_vals, cum_anom, color="#b30000", alpha=0.5, linestyle=":", label="Cumulative Anomaly")
            ax_mid_r.plot(x_vals, cum_drop, color="#d97706", alpha=0.5, linestyle=":", label="Cumulative Drop")
            ax_mid_r.plot(x_vals, cum_spike, color="#2f9e44", alpha=0.5, linestyle=":", label="Cumulative Peak")
            ax_mid_r.set_ylabel("Cumulative Alerts")
            ax_mid_r.legend(loc="upper right")

            ax_bottom = axes[2]
            summary_values = (run_intensity_smooth if use_ema_for_display else run_intensity)
            ma_run = moving_average(summary_values, moving_avg_window)
            summary_label = "Intensity (EMA)" if use_ema_for_display else "Intensity"
            ax_bottom.plot(x_vals, summary_values, color="#1b6ca8", label=summary_label)
            ax_bottom.plot(x_vals, ma_run, color="#0b7285", linestyle="--", label=f"MA({moving_avg_window})")
            if show_raw_trace and use_ema_for_display:
                ax_bottom.plot(x_vals, run_intensity, color="#1b6ca8", alpha=raw_trace_alpha, linewidth=1, label="Raw")
            anomaly_x = [i for i, flag in enumerate(run_anomaly) if flag]
            anomaly_y = [summary_values[i] for i in anomaly_x]
            if anomaly_x:
                ax_bottom.scatter(anomaly_x, anomaly_y, color="#b30000", marker="x", label="Anomaly")
            drop_x = [i for i, flag in enumerate(run_drop) if flag]
            drop_y = [summary_values[i] for i in drop_x]
            if drop_x:
                ax_bottom.scatter(drop_x, drop_y, color="#d97706", marker="v", label="Drop")
            spike_x = [i for i, flag in enumerate(run_spike) if flag]
            spike_y = [summary_values[i] for i in spike_x]
            if spike_x:
                ax_bottom.scatter(spike_x, spike_y, color="#2f9e44", marker="^", label="Peak")
            ax_bottom.set_ylabel("Intensity")
            ax_bottom.set_xlabel("Point")
            ax_bottom.grid(True, alpha=0.3)
            ax_bottom.legend(loc="upper right")

            fig_summary.suptitle("Run Summary: Health, Alerts, Efficiency")
            fig_summary.tight_layout()
            summary_plot_path = os.path.join(summary_dir, f"{log_prefix}_{run_id}_summary_plot.png")
            fig_summary.savefig(summary_plot_path, dpi=150)
            plt.close(fig_summary)
        except Exception:
            pass

    # Stage the files by type so the zip has the same folder layout as the run.
    bundle_zip_path = None
    try:
        bundle_root = os.path.join(bundle_dir, run_id)
        bundle_csv = os.path.join(bundle_root, csv_dir_name)
        bundle_snaps = os.path.join(bundle_root, snapshot_dir_name)
        bundle_summaries = os.path.join(bundle_root, summary_dir_name)
        os.makedirs(bundle_csv, exist_ok=True)
        os.makedirs(bundle_snaps, exist_ok=True)
        os.makedirs(bundle_summaries, exist_ok=True)

        if os.path.exists(log_path):
            shutil.copy2(log_path, os.path.join(bundle_csv, os.path.basename(log_path)))
        if os.path.exists(summary_path):
            shutil.copy2(summary_path, os.path.join(bundle_summaries, os.path.basename(summary_path)))
        if summary_plot_path and os.path.exists(summary_plot_path):
            shutil.copy2(summary_plot_path, os.path.join(bundle_summaries, os.path.basename(summary_plot_path)))
        for snap_path in run_snapshot_files:
            if os.path.exists(snap_path):
                shutil.copy2(snap_path, os.path.join(bundle_snaps, os.path.basename(snap_path)))

        bundle_zip_path = shutil.make_archive(bundle_root, "zip", root_dir=bundle_root)
    except Exception:
        pass

    email_status = "not_sent"
    email_error = ""
    used_config_path = refresh_email_config(os.path.join(output_dir, "email.env"))
    missing_email_vars = []
    for key in ["ASTRO_EMAIL_TO", "ASTRO_SMTP_HOST", "ASTRO_SMTP_USER", "ASTRO_SMTP_PASS"]:
        if not os.environ.get(key):
            missing_email_vars.append(key)

    if email_enabled and bundle_zip_path:
        try:
            subject = f"{email_subject_prefix} {run_id}"
            body = (
                "Attached is the run bundle with csv_logs/, snapshots/, and summaries/.\n"
                f"Run ID: {run_id}\n"
                f"Start UTC: {mission_start_utc.isoformat()}\n"
                f"End UTC: {end_time_utc.isoformat()}\n"
            )
            sent = send_email_with_attachment(email_to, subject, body, bundle_zip_path)
            email_status = "sent" if sent else "failed"
        except Exception as exc:
            email_status = "failed"
            email_error = str(exc)
    elif not email_enabled:
        email_status = "missing_config"
    elif not bundle_zip_path:
        email_status = "missing_bundle"

    if email_status == "sent" and auto_delete_bundle_after_email:
        try:
            if bundle_zip_path and os.path.exists(bundle_zip_path):
                os.remove(bundle_zip_path)
            if bundle_root and os.path.exists(bundle_root):
                shutil.rmtree(bundle_root)
        except Exception:
            pass

    try:
        if os.path.exists(summary_path):
            with open(summary_path, "a") as summary_file:
                summary_file.write("\nArtifacts\n")
                summary_file.write("-" * 40 + "\n")
                summary_file.write(f"CSV: {log_path}\n")
                if summary_plot_path:
                    summary_file.write(f"Summary plot: {summary_plot_path}\n")
                summary_file.write(f"Snapshots: {snapshot_dir}\n")
                if bundle_zip_path:
                    summary_file.write(f"Bundle zip: {bundle_zip_path}\n")
                summary_file.write(f"Email enabled: {email_enabled}\n")
                summary_file.write(f"Email status: {email_status}\n")
                if used_config_path:
                    summary_file.write(f"Email config path: {used_config_path}\n")
                if missing_email_vars:
                    summary_file.write(f"Missing email vars: {', '.join(missing_email_vars)}\n")
                if email_enabled:
                    summary_file.write(f"Email to: {email_to}\n")
                if email_error:
                    summary_file.write(f"Email error: {email_error}\n")
    except Exception:
        pass
    if save_final_snapshot:
        final_path = os.path.join(snapshot_dir, f"{log_prefix}_{run_id}_final.png")
        try:
            fig.savefig(final_path, dpi=200)
        except Exception:
            pass


animation = ani.FuncAnimation(fig, animate, interval=update_interval_ms)
try:
    plt.show()
finally:
    end_time_utc = datetime.now(pytz.utc)
    try:
        log_file.close()  # Flush the CSV before it goes into the bundle.
    finally:
        ser.close()
    save_run(end_time_utc)
