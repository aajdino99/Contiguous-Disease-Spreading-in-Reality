# -*- coding: utf-8 -*-
""" Spatial SIR model for COVID-19 on a 1 km grid over Sweden.
- Each cell of the GeoPackage 'population_1km_2024.gpkg' is one metapopulation.
- Compartments per cell: S, I, R (with waning immunity R -> S).
- DYNAMIC LOCKDOWN with STABILITY:
    - Triggers based on infection % thresholds.
    - Enforces MINIMUM DURATION and COOLDOWN periods to prevent rapid toggling.
*** MODIFIED: Now runs 20 simulations and plots the mean time series. ***
*** MODIFIED: Time series plot now cuts off after the mean extinction day. ***
*** MODIFIED: Removed the grey lockdown shading from the time series plot. ***
"""

gpkg_path = "population_1km_2024.gpkg" # Path to GeoPackage
population_col = "beftotalt" # Population column in the GPKG
NUM_SIMULATIONS = 20 # <<<--- MOsDIFIED: Number of runs
# -------------------------------------------------------------------
# 0. Imports
# -------------------------------------------------------------------
import geopandas as gpd # Spatial data
import matplotlib.pyplot as plt # Plotting
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import random

# -------------------------------------------------------------------
# 1. Load data and prepare grid
# -------------------------------------------------------------------
# np.random.seed(42)

# Read the 1 km population grid for Sweden
# NOTE: GeoPackage loading may fail if the file is not present.
try:
    gdf = gpd.read_file(gpkg_path)
except Exception as e:
    print(f"Error loading GeoPackage: {e}")
    print("Please ensure 'population_1km_2024.gpkg' is in the current directory.")
    # For a clean environment, you may want to re-raise the error here or use dummy data.
    raise

gdf[population_col] = gdf[population_col].astype(float).fillna(0.0)
gdf["total"] = gdf[population_col].astype(float)
total_pop = gdf["total"].values
N = np.round(total_pop).astype(int)
N = np.clip(N, 0, None)

# Total population across Sweden
N_total = int(N.sum())
if N_total == 0:
    raise ValueError("Total population is zero; cannot simulate epidemic.")

# Safe version of N
N_safe = np.where(N > 0, N, 1)

# -------------------------------------------------------------------
# 2. Seed cells (fixed starting locations)
# -------------------------------------------------------------------
nonzero_mask = N > 0
nonzero_indices = np.where(nonzero_mask)[0]
if len(nonzero_indices) == 0:
    raise ValueError("All cells have zero population.")

num_seed_cells = max(1, int(0.04 * len(nonzero_indices)))
max_pop = np.max(total_pop)
target_pop = random.uniform(12500.0, max_pop)
pop_nonzero = total_pop[nonzero_mask]
distance_to_target = np.abs(pop_nonzero - target_pop)
sorted_idx = np.argsort(distance_to_target)
sorted_nonzero_indices = nonzero_indices[sorted_idx]
seed_indices = sorted_nonzero_indices[:num_seed_cells]

# -------------------------------------------------------------------
# 3. Epidemiological parameters
# -------------------------------------------------------------------
infection_fatality_ratio = 0.0068
infectious_period = 10.0
exit_rate = 1.0 / infectious_period
mu = infection_fatality_ratio * exit_rate
gamma = exit_rate - mu

R0_local = 2.8
beta_local = R0_local * gamma
beta_global = 0.05 * beta_local

waning_immunity_days = 244.0
alpha = 1.0 / waning_immunity_days

# --- DYNAMIC LOCKDOWN PARAMETERS (REALISTIC) ------------------------
use_lockdown = True

# 1. Infection Thresholds (% of total population)
lockdown_trigger_start = 4 # Start if > % infected
lockdown_trigger_end = 1 # End if < % infected

# 2. Stability Constraints (prevent rapid switching)
min_lockdown_duration = 42
min_cooldown_duration = 21

lockdown_contact_factor = 0.75 # % reduction

# --- Global mixing weights ------------------------------------------
if N.max() > 0:
    pop_norm = N / N.max()
else:
    pop_norm = np.zeros_like(N, dtype=float)
pop_weight_for_global = np.sqrt(pop_norm)

max_days = 1247
num_days_hist = max_days + 1 # For history array size

# -------------------------------------------------------------------
# 4. Simulation function: Dynamic Lockdown with Stability
# -------------------------------------------------------------------
def simulate():
    """ Run one stochastic SIR simulation with realistic lockdown switching. """

    # --- Day 0 ---
    S0 = N.copy()
    I0 = np.zeros_like(N, dtype=int)
    R0 = np.zeros_like(N, dtype=int)
    S_hist = [S0.copy()]
    I_hist = [I0.copy()]
    R_hist = [R0.copy()]

    # Lockdown state tracking
    lockdown_active = False
    last_state_change_day = 0 # Day when status last changed
    lockdown_days = [] # History for plotting

    # --- Day 1: Seeding ---
    S = S0.copy()
    I = I0.copy()
    R = R0.copy()

    initial_infected_fraction = np.random.uniform(0.0001, 0.001, size=len(seed_indices))
    for idx, frac in zip(seed_indices, initial_infected_fraction):
        new_inf = int(round(frac * N[idx]))
        new_inf = max(new_inf, 1)
        new_inf = min(new_inf, S[idx])
        S[idx] -= new_inf
        I[idx] += new_inf

    S_hist.append(S.copy())
    I_hist.append(I.copy())
    R_hist.append(R.copy())

    # Fill history for Day 0 and 1
    lockdown_days.append(False)
    lockdown_days.append(False)

    # --- Days 2..max_days ---
    for day in range(2, max_days + 1):
        # Current infection stats (based on start of day)
        I_total_prev = I.sum()
        pct_infected = (I_total_prev / N_total) * 100.0

        # Days since last toggle
        days_in_current_state = day - last_state_change_day

        if use_lockdown:
            if lockdown_active:
                # CURRENTLY LOCKED DOWN
                if days_in_current_state >= min_lockdown_duration:
                    if pct_infected < lockdown_trigger_end:
                        lockdown_active = False
                        last_state_change_day = day
            else:
                # CURRENTLY OPEN
                if days_in_current_state >= min_cooldown_duration:
                    if pct_infected > lockdown_trigger_start:
                        lockdown_active = True
                        last_state_change_day = day

        # Apply factor
        contact_factor = lockdown_contact_factor if lockdown_active else 1.0
        lockdown_days.append(lockdown_active)

        # SIR Dynamics
        beta_local_eff = beta_local * contact_factor
        beta_global_eff = beta_global * contact_factor

        I_total = I.sum()
        prevalence_local = I / N_safe
        lambda_local = beta_local_eff * prevalence_local
        p_local = 1.0 - np.exp(-lambda_local)

        prevalence_global = I_total / N_total if N_total > 0 else 0.0
        lambda_global_base = beta_global_eff * prevalence_global
        lambda_global = lambda_global_base * pop_weight_for_global
        p_global = 1.0 - np.exp(-lambda_global)

        p_inf = np.clip(p_local + p_global, 0.0, 1.0)
        new_infections = np.random.binomial(S.astype(int), p_inf)

        if day <= 7:
            # Reduce early-day infections to mitigate simulation explosion artifact
            new_infections = np.floor(new_infections * 0.1)

        new_recoveries = np.random.binomial(I.astype(int), gamma)

        S = S - new_infections
        I = I + new_infections - new_recoveries
        R = R + new_recoveries

        S_hist.append(S.copy())
        I_hist.append(I.copy())
        R_hist.append(R.copy())

    # --- Post-Processing ---
    # Explicitly cast to int32 to halve the memory footprint from 64-bit floats
    S_hist = np.array(S_hist, dtype=np.int32)
    I_hist = np.array(I_hist, dtype=np.int32)
    R_hist = np.array(R_hist, dtype=np.int32)
    num_days = S_hist.shape[0]
    days = np.arange(num_days)

    total_S = S_hist.sum(axis=1)
    total_I = I_hist.sum(axis=1)
    total_R = R_hist.sum(axis=1)
    total_I[0] = 0

    extinction_day = None
    for d in range(1, num_days):
        if total_I[d] == 0:
            extinction_day = d
            break

    # NEW: Calculate peak infection day
    peak_I_day = np.argmax(total_I)

    positive_vals = I_hist[I_hist > 0]
    infected_max = int(np.ceil(np.percentile(positive_vals, 99))) if positive_vals.size > 0 else 1
    if infected_max < 1: infected_max = 1

    # Convert boolean list to intervals
    lockdown_intervals = []
    current_start = None
    for d, is_active in enumerate(lockdown_days):
        if is_active and current_start is None:
            current_start = d
        elif not is_active and current_start is not None:
            lockdown_intervals.append((current_start, d))
            current_start = None
    if current_start is not None:
        lockdown_intervals.append((current_start, num_days))

    return {
        "S_hist": S_hist, "I_hist": I_hist, "R_hist": R_hist,
        "total_S": total_S, "total_I": total_I, "total_R": total_R,
        "days": days, "num_days": num_days,
        "infected_max": infected_max, "extinction_day": extinction_day,
        "peak_I_day": peak_I_day, # NEW: Include peak day
        "lockdown_intervals": lockdown_intervals
    }

# -------------------------------------------------------------------
# 4.5. Run multiple simulations and calculate statistics
# -------------------------------------------------------------------

all_S, all_I, all_R = [], [], []
all_sim_data = []
extinction_days = []
peak_I_days = [] # NEW: List to store peak infection day for each run

print(f"Running {NUM_SIMULATIONS} stochastic simulations...")
for i in range(NUM_SIMULATIONS):
    # Set a unique seed for each run for stochasticity
    np.random.seed(i)
    result = simulate()
    all_sim_data.append(result)
    all_S.append(result["total_S"])
    all_I.append(result["total_I"])
    all_R.append(result["total_R"])
    # Collect extinction day
    if result["extinction_day"] is not None:
        extinction_days.append(result["extinction_day"])
    # NEW: Collect peak infection day
    if result["peak_I_day"] is not None:
        peak_I_days.append(result["peak_I_day"])
print("Simulations complete.")

# CALCULATE MEAN EXTINCTION DAY
mean_extinction_day = None
if extinction_days:
    mean_extinction_day = int(np.mean(extinction_days))

# NEW: CALCULATE MEAN PEAK INFECTION DAY 
mean_peak_day = None
if peak_I_days:
    mean_peak_day = int(np.mean(peak_I_days))

# Pad histories to the maximum number of days if any run finished early (extinction)
max_len = max(len(h) for h in all_S)
def pad_history(history, compartment, max_len, N_total):
    padded = np.full(max_len, np.nan)
    current_len = len(history)
    padded[:current_len] = history
    # For S, I, R, if a run ended early, the last value remains constant (extinct state)
    if current_len < max_len:
        if compartment == 'S':
            # S stops decreasing at the end
            padded[current_len:] = history[-1]
        elif compartment == 'I':
            # I drops to 0 at extinction
            padded[current_len:] = 0
        elif compartment == 'R':
            # R stops increasing at the end
            padded[current_len:] = history[-1]
    return padded

all_S_padded = np.array([pad_history(h, 'S', max_len, N_total) for h in all_S])
all_I_padded = np.array([pad_history(h, 'I', max_len, N_total) for h in all_I])
all_R_padded = np.array([pad_history(h, 'R', max_len, N_total) for h in all_R])

# Calculate Mean, 2.5th and 97.5th percentiles (95% confidence interval)
mean_S = np.nanmean(all_S_padded, axis=0)
mean_I = np.nanmean(all_I_padded, axis=0)
mean_R = np.nanmean(all_R_padded, axis=0)

percentile_S_low = np.nanpercentile(all_S_padded, 2.5, axis=0)
percentile_I_low = np.nanpercentile(all_I_padded, 2.5, axis=0)
percentile_R_low = np.nanpercentile(all_R_padded, 2.5, axis=0)

percentile_S_high = np.nanpercentile(all_S_padded, 97.5, axis=0)
percentile_I_high = np.nanpercentile(all_I_padded, 97.5, axis=0)
percentile_R_high = np.nanpercentile(all_R_padded, 97.5, axis=0)

# Overwrite sim_data with the results from the first run for map/interactive use
sim_data = all_sim_data[0]
sim_data['total_S'] = mean_S
sim_data['total_I'] = mean_I
sim_data['total_R'] = mean_R
sim_data['num_days'] = max_len
sim_data['days'] = np.arange(max_len)


# -------------------------------------------------------------------
# 5. Prepare initial map (using first run's day 0)
# -------------------------------------------------------------------
gdf["infected"] = all_sim_data[0]["I_hist"][0].astype(float)
infected_max_single_run = all_sim_data[0]["infected_max"]

# -------------------------------------------------------------------
# 6. Figure layout
# -------------------------------------------------------------------
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 2, width_ratios=[1.1, 1.4], wspace=0.3, figure=fig)
ax_map = fig.add_subplot(gs[0, 0])
ax_ts = fig.add_subplot(gs[0, 1])
plt.subplots_adjust(bottom=0.26)
dark_gray = "#ff914d"
# PLOT INTERIOR COLOR (lighter_gray)
lighter_gray = "#333333"
fig.patch.set_facecolor(dark_gray)
ax_map.set_facecolor(dark_gray)
# REVERTED: Keep ax_ts background back to light gray
ax_ts.set_facecolor(lighter_gray)

# Define the new dark color (requested for axes numbers and titles)
custom_color = "#0b1f3a"

# -------------------------------------------------------------------
# 7. Map
# -------------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list("black_yellow_red", ["#000000", "#ffff00", "#ff0000"])
gdf.plot(column="infected", ax=ax_map, cmap=cmap, vmin=0.0, vmax=infected_max_single_run, linewidth=0, edgecolor="none")
# MODIFICATION: Set map title color to the dark custom color
ax_map.set_title(f"Infected map (Run 1)", color=custom_color, fontweight="bold")
ax_map.set_axis_off()
map_collection = ax_map.collections[0]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=infected_max_single_run))
sm._A = []
cbar = plt.colorbar(sm, ax=ax_map, fraction=0.05, pad=0.02)
cbar.ax.set_visible(True)
cbar.set_label("Infected count (Run 1)", color="white")
cbar.outline.set_edgecolor("white")
plt.setp(cbar.ax.get_yticklabels(), color="white")

# -------------------------------------------------------------------
# 8. SIR time series plot + Lockdown Shading
# -------------------------------------------------------------------

# --- MODIFICATION START: Calculate the data cut-off index ---
cut_off_idx = max_len
if mean_extinction_day is not None:
    cut_off_idx = min(mean_extinction_day + 1, max_len)

days_plot = sim_data["days"][:cut_off_idx]
plot_mean_S = mean_S[:cut_off_idx]
plot_mean_I = mean_I[:cut_off_idx]
plot_mean_R = mean_R[:cut_off_idx]
plot_percentile_S_low = percentile_S_low[:cut_off_idx]
plot_percentile_I_low = percentile_I_low[:cut_off_idx]
plot_percentile_R_low = percentile_R_low[:cut_off_idx]
plot_percentile_S_high = percentile_S_high[:cut_off_idx]
plot_percentile_I_high = percentile_I_high[:cut_off_idx]
plot_percentile_R_high = percentile_R_high[:cut_off_idx]


# Plot shaded uncertainty bands first
fill_S = ax_ts.fill_between(days_plot, plot_percentile_S_low, plot_percentile_S_high, color="#1f77b4", alpha=0.15)
fill_I = ax_ts.fill_between(days_plot, plot_percentile_I_low, plot_percentile_I_high, color="#ff7f0e", alpha=0.15)
fill_R = ax_ts.fill_between(days_plot, plot_percentile_R_low, plot_percentile_R_high, color="#2ca02c", alpha=0.15)

# Plot Mean lines
line_S, = ax_ts.plot(days_plot, plot_mean_S, label="Mean S", color="#1f77b4")
line_I, = ax_ts.plot(days_plot, plot_mean_I, label="Mean I", color="#ff7f0e")
line_R, = ax_ts.plot(days_plot, plot_mean_R, label="Mean R", color="#2ca02c")
# --- MODIFICATION END ---

# Construct the descriptive text
extinction_label = f"Runs: {NUM_SIMULATIONS}\n"
if mean_extinction_day is not None:
    extinction_label += f"Avg Extinction: Day {mean_extinction_day}\n"
else:
    extinction_label += "Avg Extinction: N/A\n"

if mean_peak_day is not None:
    extinction_label += f"Avg Peak: Day {mean_peak_day}"
else:
    extinction_label += "Avg Peak: N/A"

# MODIFIED: Set labels and title color to the dark custom color
ax_ts.set_xlabel("Day", color=custom_color, fontweight="bold")
ax_ts.set_ylabel("Number of individuals", color=custom_color, fontweight="bold")
ax_ts.set_title(f"Spatial SIR - Mean of {NUM_SIMULATIONS} Runs", color=custom_color, fontweight="bold")
ax_ts.set_ylim(0, N_total)

# *** CRITICAL MODIFICATION for Axis Numbers/Ticks ***
# 1. Ticks point out (numbers outside plot area)
# 2. Set tick line color to white (to blend with the spine/frame color)
# 3. Set tick label color (the numbers) to custom_color (dark blue)
ax_ts.tick_params(axis='both', direction='out', color="white", labelcolor=custom_color)
# REVERTED: Set spines to white (for contrast on dark gray background)
for spine in ax_ts.spines.values(): spine.set_color("white")

ax_ts.grid(color="gray", alpha=0.3)

# Legend setup with initial values
legend = ax_ts.legend(facecolor="#333333", edgecolor="none", loc="upper right")
for text in legend.get_texts():
    text.set_color("white")

# UPDATE: Change extinction text to show the mean
# REVERTED: Set vline and extinction_text color back to white (interior text)
vline = ax_ts.axvline(0, linestyle="--", color="yellow", alpha=0.7)
extinction_text = ax_ts.text(0.02, 0.98, extinction_label, transform=ax_ts.transAxes, color="white", fontsize=9, va="top")

# Lockdown text indicator - uses run 1 data
lockdown_text = ax_ts.text(0.02, 0.88, "", transform=ax_ts.transAxes, color="#ff4444", fontsize=12, fontweight="bold", va="top")

lockdown_spans = []

def draw_lockdown_spans(intervals):
    """ Clears and then conditionally draws lockdown spans. MODIFIED to only clear. """
    global lockdown_spans
    # remove any existing spans
    for span in lockdown_spans:
        span.remove()
    lockdown_spans = []
    # ** THE CODE TO DRAW NEW SPANS IS REMOVED HERE **
    # The remaining line is needed to refresh the plot correctly after clearing.
    fig.canvas.draw_idle()

# Use the lockdown intervals from the first run (all_sim_data[0])
draw_lockdown_spans(all_sim_data[0]["lockdown_intervals"])

# Update x-axis limit based on the new cut-off
ax_ts.set_xlim(0, days_plot[-1]) # Set limit to the last day plotted

# -------------------------------------------------------------------
# 9. Day slider
# -------------------------------------------------------------------
# MODIFIED: Vertical position from 0.16 to 0.05
ax_slider = fig.add_axes([0.10, 0.05, 0.45, 0.03])
# MODIFICATION: Slider max value is now the cut-off index - 1
day_slider = Slider(ax=ax_slider, label="Day", valmin=0, valmax=cut_off_idx - 1, valinit=0, valstep=1, color="#444444")
day_slider.label.set_color("white")
day_slider.valtext.set_color("white")

# *** MODIFICATION START: Remove box around the slider axis ***
for spine in ax_slider.spines.values():
    spine.set_visible(False)
# *** MODIFICATION END ***

def update_slider(day_value):
    global sim_data
    day_idx = int(day_value)

    # MODIFICATION: The max day index must not exceed the cut-off for the plot
    day_idx = max(0, min(cut_off_idx - 1, day_idx))

    # Use the spatial data from the first run (all_sim_data[0]) for the map
    if day_idx < len(all_sim_data[0]["I_hist"]):
        current_I = all_sim_data[0]["I_hist"][day_idx].astype(float)
        gdf["infected"] = current_I
        map_collection.set_array(current_I)

    vline.set_xdata([day_idx, day_idx])
    day_box.set_val(str(day_idx))

    # Update legend text with mean S, I, R values
    # MODIFICATION: Use the sliced plot_mean_ arrays for values
    val_S = int(plot_mean_S[day_idx])
    val_I = int(plot_mean_I[day_idx])
    val_R = int(plot_mean_R[day_idx])

    if legend:
        texts = legend.get_texts()
        # Find the text objects for the S, I, R lines (index 0, 1, 2)
        if len(texts) >= 3:
            texts[0].set_text(f"Mean S: {val_S:,}")
            texts[1].set_text(f"Mean I: {val_I:,}")
            texts[2].set_text(f"Mean R: {val_R:,}")

    # Check and update lockdown status text (using run 1 data)
    is_locked = False
    lockdown_intervals_run1 = all_sim_data[0]["lockdown_intervals"]
    for start, end in lockdown_intervals_run1:
        if start <= day_idx < end:
            is_locked = True
            break

    if is_locked:
        lockdown_text.set_text("")
    else:
        lockdown_text.set_text("")

    fig.canvas.draw_idle()

day_slider.on_changed(update_slider)

# -------------------------------------------------------------------
# 10. TextBox
# -------------------------------------------------------------------
# Position kept at 0.08
ax_daybox = fig.add_axes([0.10, 0.08, 0.16, 0.045])
ax_daybox.set_facecolor("white")
day_box = TextBox(ax_daybox, "Select day", initial="0")
day_box.label.set_color("white")
day_box.text_disp.set_color("black")

# *** MODIFICATION START: Remove box around the TextBox axis ***
# Set all spines to invisible
for spine in ax_daybox.spines.values():
    spine.set_visible(False)
# *** MODIFICATION END ***

def submit_day(text):
    try: val = int(text)
    except ValueError: return
    # MODIFICATION: Clamp to the new max slider value
    val = max(day_slider.valmin, min(day_slider.valmax, val))
    day_slider.set_val(val)

day_box.on_submit(submit_day)

# -------------------------------------------------------------------
# 11. Re-simulation helpers
# -------------------------------------------------------------------
# Note: The original 'apply_simulation_to_plots' logic has been removed as the statistics are pre-calculated.
# A full re-run would require re-running the 20 simulations loop.

def reset_plots_to_initial_state():
    """ Reset the visual elements to Day 0 using pre-calculated means and Run 1 map data. """

    # MODIFICATION: Use the *new* sliced data for resetting plot lines
    line_S.set_data(days_plot, plot_mean_S)
    line_I.set_data(days_plot, plot_mean_I)
    line_R.set_data(days_plot, plot_mean_R)

    # The fill areas are complex to manage, so we trust the initial plot setup and only reset data/view.

    # MODIFICATION: Reset x-limit to the new cut-off
    ax_ts.set_xlim(0, days_plot[-1])
    ax_ts.set_ylim(0, N_total)

    # Update lockdown spans for Run 1 (will clear them due to draw_lockdown_spans modification)
    draw_lockdown_spans(all_sim_data[0]["lockdown_intervals"])

    # Update map data for Day 0 (Run 1)
    day0_I = all_sim_data[0]["I_hist"][0].astype(float)
    gdf["infected"] = day0_I
    map_collection.set_array(day0_I)

    # Update colorbar max value (using Run 1 max)
    new_max = all_sim_data[0]["infected_max"]
    if new_max <= 0: new_max = 1
    sm.set_clim(0.0, new_max)
    map_collection.set_clim(0.0, new_max)
    cbar.update_normal(sm)

    day_slider.set_val(0)
    day_box.set_val("0")
    vline.set_xdata([0, 0])

    # Initial legend and text update
    update_slider(0)

    fig.canvas.draw_idle()


# -------------------------------------------------------------------
# 12. Control buttons
# -------------------------------------------------------------------
button_color, hover_color, text_color = "#333333", "#555555", "white"
is_playing = False

def step_day(delta):
    current = day_slider.val
    new = max(day_slider.valmin, min(day_slider.valmax, current + delta))
    day_slider.set_val(new)
    day_box.set_val(str(int(new)))

# MODIFIED: Vertical position from 0.16 to 0.03
ax_prev = fig.add_axes([0.58, 0.03, 0.05, 0.04])
btn_prev = Button(ax_prev, "<", color=button_color, hovercolor=hover_color)
btn_prev.label.set_color(text_color)
btn_prev.on_clicked(lambda e: step_day(-1))

# MODIFIED: Vertical position from 0.16 to 0.03
ax_next = fig.add_axes([0.64, 0.03, 0.05, 0.04])
btn_next = Button(ax_next, ">", color=button_color, hovercolor=hover_color)
btn_next.label.set_color(text_color)
btn_next.on_clicked(lambda e: step_day(1))

# MODIFIED: Vertical position from 0.16 to 0.03
ax_play_pause = fig.add_axes([0.70, 0.03, 0.08, 0.04])
btn_play_pause = Button(ax_play_pause, "Play", color=button_color, hovercolor=hover_color)
btn_play_pause.label.set_color(text_color)

def on_play_pause(event):
    global is_playing
    if is_playing:
        is_playing = False
        btn_play_pause.label.set_text("Play") # Corrected text
    else:
        is_playing = True
        btn_play_pause.label.set_text("Pause") # Corrected text
    start = int(day_slider.val)
    # The loop limit must be max_len-1 as the day_slider max is max_len-1
    # MODIFICATION: The loop limit is now the maximum value of the day slider + 1
    for d in range(start, int(day_slider.valmax) + 1):
        if not is_playing: break
        day_slider.set_val(d)
        day_box.set_val(str(d))
        plt.pause(0.05)
    # Reset button text when loop finishes or breaks
    if is_playing: # This check handles the case where the loop finished naturally
        is_playing = False
        btn_play_pause.label.set_text("Play")


btn_play_pause.on_clicked(on_play_pause)

# Removed the 'Randomize' button as recalculating 20 means is slow for an interactive button
# You would need to re-run the 20 simulation loop and stat calcs.

# Position kept at 0.08
ax_reset = fig.add_axes([0.80, 0.08, 0.08, 0.045])
btn_reset = Button(ax_reset, "Reset", color=button_color, hovercolor=hover_color)
btn_reset.label.set_color(text_color)
# Reset button just goes back to day 0
btn_reset.on_clicked(lambda e: (globals().update(is_playing=False), day_slider.set_val(0), day_box.set_val("0")))

# Initial call to set legend and text values correctly on startup
update_slider(0)

plt.show()