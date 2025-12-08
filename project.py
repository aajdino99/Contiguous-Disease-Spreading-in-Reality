# -*- coding: utf-8 -*- 
"""
Spatial SIR model for COVID‑19 on a 1 km grid over Sweden.
- Each cell of the GeoPackage 'population_1km_2024.gpkg' is one metapopulation.
- Compartments per cell: S, I, R (with waning immunity R -> S).
- Transmission is a mix of local (within-cell) and weak global (between-cell) contacts.
- A simple "lockdown" period scales down the contact rate.

The epidemiological parameters below are chosen to approximate ancestral (pre‑variant) SARS‑CoV‑2 dynamics:
R0 ~ 2.8 (global early-pandemic meta-analyses)
IFR ~ 0.68% (infection fatality ratio, population average)
Infectious period ~ 7 days
Immunity duration ~ 365 days

These are supported by the literature and explained in the report.
"""

gpkg_path = "population_1km_2024.gpkg"  # Path to GeoPackage
population_col = "beftotalt"  # Population column in the GPKG

# -------------------------------------------------------------------
# 0. Imports
# -------------------------------------------------------------------
import geopandas as gpd  # Spatial data
import matplotlib.pyplot as plt  # Plotting
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import random

# -------------------------------------------------------------------
# 1. Load data and prepare grid
# -------------------------------------------------------------------
# NOTE: we do NOT fix a random seed, so each "Randomize" run is different.
# You can uncomment the next line to make results reproducible.
# np.random.seed(42)

# Read the 1 km population grid for Sweden
gdf = gpd.read_file(gpkg_path)

# Make sure the population column is numeric and replace NaNs by 0
gdf[population_col] = gdf[population_col].astype(float).fillna(0.0)

# Store the population in a convenient column
gdf["total"] = gdf[population_col].astype(float)  # Population per cell as integers

total_pop = gdf["total"].values
N = np.round(total_pop).astype(int)  # Number of grid cells
num_cells = len(gdf)

# Clip any negative populations (should not happen, but be safe)
N = np.clip(N, 0, None)

# Total population across Sweden (used in global mixing)
N_total = int(N.sum())
if N_total == 0:
    raise ValueError("Total population is zero; cannot simulate epidemic.")

# Safe version of N to avoid division by zero in I/N
N_safe = np.where(N > 0, N, 1)

# -------------------------------------------------------------------
# 2. Seed cells (fixed starting locations)
# - Choose ~4% of non-zero cells around a target population (e.g. cities)
# - In those cells, infect 1–5% of people on day 1
# -------------------------------------------------------------------
nonzero_mask = N > 0
nonzero_indices = np.where(nonzero_mask)[0]
if len(nonzero_indices) == 0:
    raise ValueError("All cells have zero population; cannot simulate epidemic.")

# Number of seed cells (~4% of all non-empty cells, at least 1)
num_seed_cells = max(1, int(0.04 * len(nonzero_indices)))

# Randomly choose a target cell population between 12,500 and the maximum
max_pop = np.max(total_pop)
target_pop = random.uniform(12500.0, max_pop)

# Distance of each non-zero cell's population to the target pop
pop_nonzero = total_pop[nonzero_mask]
distance_to_target = np.abs(pop_nonzero - target_pop)

# Sort cells by how close they are to the target population
sorted_idx = np.argsort(distance_to_target)
sorted_nonzero_indices = nonzero_indices[sorted_idx]

# Final list of seed cell indices (fixed set for the whole session)
seed_indices = sorted_nonzero_indices[:num_seed_cells]

# -------------------------------------------------------------------
# 3. Epidemiological parameters (COVID‑19: ancestral strain)
# -------------------------------------------------------------------
# All parameters are per day and refer to a simple SIR model with
# waning immunity and optional lockdown.
# --- Disease severity: IFR and infectious period --------------------
# Population-average infection fatality ratio (IFR) for COVID‑19
# Meyerowitz‑Katz & Merone (2020, Int J Infect Dis) meta-analysis:
# IFR ≈ 0.68% (0.0068 as probability)
infection_fatality_ratio = 0.0068

# Mean infectious period (duration of being contagious), in days.
# Scoping reviews of viral shedding & infectiousness suggest 7–9 days;
# we use 7 days as a simple, widely used assumption.
infectious_period = 7.0  # days

# People leave the infectious compartment (by recovery or death) at rate: exit_rate
exit_rate = 1.0 / infectious_period  # ≈ 0.1429 / day

# Split exits into recovery so that IFR = mu / gamma
mu = infection_fatality_ratio * exit_rate  # death hazard per day
gamma = exit_rate - mu  # recovery hazard per day

# --- Transmission: R0 and beta --------------------------------------
# Basic reproduction number R0 for early COVID‑19 (ancestral strain),
# pooled estimate ≈ 2.8–3.0 in meta-analyses and Western Europe models.
R0_local = 2.8
beta_local = R0_local * gamma  # ≈ 0.40 per day

# Small global mixing component: long-distance travel, commuting, etc.
# Kept at 5% of local beta so that most infections are local.
beta_global = 0.05 * beta_local

# --- Waning immunity: R -> S ----------------------------------------
# Typical duration of protection after infection (pre-Omicron) is
# of order one year or more; we choose 365 days.
waning_immunity_days = 244.0
alpha = 1.0 / waning_immunity_days  # probability per day that R -> S

# --- Lockdown / NPI parameters --------------------------------------
# use_lockdown : whether we apply a period of reduced contacts
# lockdown_start_day : first day with reduced contacts
# lockdown_duration : length of that period in days
# lockdown_contact_factor : multiplier for beta during lockdown
#
# Literature suggests strict combined NPIs can reduce R by ~60–80%.
# We choose 0.35 by default (about 65% contact reduction). A value of
# 0.25 would represent a strict lockdown; 0.6–0.7 a Swedish-style mild NPI.
use_lockdown = True
lockdown_start_day = 15  # e.g. day 60 after simulation start
lockdown_duration = 120  # ~4 months
lockdown_contact_factor = 0.7  # 65% reduction in contacts

# --- Global mixing weights (population dependent) -------------------
# Normalize population by its maximum to [0,1] if N.max() > 0:
if N.max() > 0:
    pop_norm = N / N.max()
else:
    pop_norm = np.zeros_like(N, dtype=float)

# Global mixing weight (large cities have stronger global coupling)
pop_weight_for_global = np.sqrt(pop_norm)

# Simulation horizon in days (0..max_days)
max_days = 1247

# -------------------------------------------------------------------
# 4. Simulation function: one stochastic SIR realization
# -------------------------------------------------------------------
def simulate():
    """ Run one stochastic SIR simulation over all cells and all days.
    Returns a dictionary with:
    S_hist, I_hist, R_hist : (day, cell) arrays
    total_S, total_I, total_R : totals over Sweden per day
    days : array [0..max_days]
    num_days : max_days + 1
    infected_max : 99th percentile of infected count-s for map scaling
    extinction_day: first day when total_I returns to 0 (if any)
    """
    # --- Day 0: everyone susceptible, no infection -------------------
    S0 = N.copy()
    I0 = np.zeros_like(N, dtype=int)
    R0 = np.zeros_like(N, dtype=int)
    S_hist = [S0.copy()]
    I_hist = [I0.copy()]
    R_hist = [R0.copy()]

    # --- Day 1: seed infections in the chosen high-population cells --
    S = S0.copy()
    I = I0.copy()
    R = R0.copy()

    # Start with a very low initial infected fraction (e.g., 0.001 for 0.1%)
    initial_infected_fraction = np.random.uniform(0.0001, 0.001, size=len(seed_indices))
    for idx, frac in zip(seed_indices, initial_infected_fraction):
        new_inf = int(round(frac * N[idx]))  # infect a small number
        new_inf = max(new_inf, 1)  # at least 1 if possible
        new_inf = min(new_inf, S[idx])  # not more than susceptibles
        S[idx] -= new_inf
        I[idx] += new_inf

    S_hist.append(S.copy())
    I_hist.append(I.copy())
    R_hist.append(R.copy())

    # --- Days 2..max_days: SIR dynamics -----------------------------
    for day in range(2, max_days + 1):
        # Default contacts (no lockdown)
        contact_factor = 1.0

        # Reduce contacts during lockdown period
        if use_lockdown:
            if lockdown_start_day <= day < lockdown_start_day + lockdown_duration:
                contact_factor = lockdown_contact_factor

        # Effective transmission rates under current interventions
        beta_local_eff = beta_local * contact_factor
        beta_global_eff = beta_global * contact_factor

        # Total infected in Sweden
        I_total = I.sum()

        # Local prevalence I/N per cell
        prevalence_local = I / N_safe
        lambda_local = beta_local_eff * prevalence_local
        p_local = 1.0 - np.exp(-lambda_local)

        # Global prevalence and intensity
        prevalence_global = I_total / N_total if N_total > 0 else 0.0
        lambda_global_base = beta_global_eff * prevalence_global
        lambda_global = lambda_global_base * pop_weight_for_global
        p_global = 1.0 - np.exp(-lambda_global)

        # Total infection probability per cell
        p_inf = np.clip(p_local + p_global, 0.0, 1.0)

        # New infections: S -> I
        new_infections = np.random.binomial(S.astype(int), p_inf)

        # Gradually increase the infection intensity by scaling up in early days
        if day <= 7:
            new_infections = np.floor(new_infections * 0.1)  # scale down for first 7 days

        # Recoveries: I -> R
        new_recoveries = np.random.binomial(I.astype(int), gamma)

        # Update compartments
        S = S - new_infections
        I = I + new_infections - new_recoveries
        R = R + new_recoveries

        S_hist.append(S.copy())
        I_hist.append(I.copy())
        R_hist.append(R.copy())

    # Convert histories to arrays
    S_hist = np.array(S_hist)
    I_hist = np.array(I_hist)
    R_hist = np.array(R_hist)

    num_days = S_hist.shape[0]
    days = np.arange(num_days)

    total_S = S_hist.sum(axis=1)
    total_I = I_hist.sum(axis=1)
    total_R = R_hist.sum(axis=1)

    # Ensure total_I[0] is exactly 0
    total_I[0] = 0

    # Extinction day: first day after day 0 with total_I == 0
    extinction_day = None
    for d in range(1, num_days):
        if total_I[d] == 0:
            extinction_day = d
            break

    # For map color scaling: 99th percentile of positive infected counts
    positive_vals = I_hist[I_hist > 0]
    if positive_vals.size > 0:
        infected_max = int(np.ceil(np.percentile(positive_vals, 99)))
    else:
        infected_max = 1
    if infected_max < 1:
        infected_max = 1

    return {
        "S_hist": S_hist,
        "I_hist": I_hist,
        "R_hist": R_hist,
        "total_S": total_S,
        "total_I": total_I,
        "total_R": total_R,
        "days": days,
        "num_days": num_days,
        "infected_max": infected_max,
        "extinction_day": extinction_day,
    }

# Run one simulation to initialize plots
sim_data = simulate()

# -------------------------------------------------------------------
# 5. Prepare initial map (day 0 infections = 0)
# -------------------------------------------------------------------
gdf["infected"] = sim_data["I_hist"][0].astype(float)

# -------------------------------------------------------------------
# 6. Figure layout with GridSpec (map + time series)
# -------------------------------------------------------------------
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 2, width_ratios=[1.1, 1.4], wspace=0.3, figure=fig)
ax_map = fig.add_subplot(gs[0, 0])
ax_ts = fig.add_subplot(gs[0, 1])
plt.subplots_adjust(bottom=0.26)
dark_gray = "#222222"
lighter_gray = "#333333"
fig.patch.set_facecolor(dark_gray)
ax_map.set_facecolor(dark_gray)
ax_ts.set_facecolor(lighter_gray)

# -------------------------------------------------------------------
# 7. Map: black → yellow → red colormap + colorbar
# -------------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list("black_yellow_red", ["#000000", "#ffff00", "#ff0000"])
infected_max = sim_data["infected_max"]
gdf.plot(column="infected", ax=ax_map, cmap=cmap, vmin=0.0, vmax=infected_max, linewidth=0, edgecolor="none")
ax_map.set_title("Infected map", color="white")
ax_map.set_axis_off()
map_collection = ax_map.collections[0]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=infected_max))
sm._A = []
cbar = plt.colorbar(sm, ax=ax_map, fraction=0.05, pad=0.02)
cbar.set_label("Infected count", color="white")
cbar.outline.set_edgecolor("white")
plt.setp(cbar.ax.get_yticklabels(), color="white")

# -------------------------------------------------------------------
# 8. SIR time series plot + lockdown shading
# -------------------------------------------------------------------
line_S, = ax_ts.plot(sim_data["days"], sim_data["total_S"], label="S", color="#1f77b4")
line_I, = ax_ts.plot(sim_data["days"], sim_data["total_I"], label="I", color="#ff7f0e")
line_R, = ax_ts.plot(sim_data["days"], sim_data["total_R"], label="R", color="#2ca02c")
ax_ts.set_xlabel("Day", color="white")
ax_ts.set_ylabel("Number of individuals", color="white")
ax_ts.set_title("Spatial SIR model for COVID‑19", color="white")
ax_ts.set_ylim(0, N_total)  # Adjust xlim based on extinction day
extinction_day = sim_data["extinction_day"]
ax_ts.set_xlim(0, extinction_day if extinction_day is not None else max_days)
ax_ts.tick_params(colors="white")
for spine in ax_ts.spines.values():
    spine.set_color("white")
ax_ts.grid(color="gray", alpha=0.3)

# Shade lockdown period in gray if used
if use_lockdown:
    ax_ts.axvspan(lockdown_start_day, lockdown_start_day + lockdown_duration, color="gray", alpha=0.15)

legend = ax_ts.legend(facecolor="#333333", edgecolor="none")
for text in legend.get_texts():
    text.set_color("white")

# Vertical line = currently selected day
vline = ax_ts.axvline(0, linestyle="--", color="white", alpha=0.7)

def format_extinction(ext_day):
    if ext_day is None:
        return "Extinction: > 1247"
    else:
        return f"Extinction day: {ext_day}"

extinction_text = ax_ts.text(0.02, 0.95, format_extinction(sim_data["extinction_day"]), transform=ax_ts.transAxes, color="white", fontsize=9, va="top")

# -------------------------------------------------------------------
# 9. Day slider (0..max_days)
# -------------------------------------------------------------------
ax_slider = fig.add_axes([0.10, 0.16, 0.45, 0.03])
day_slider = Slider(ax=ax_slider, label="Day", valmin=0, valmax=max_days, valinit=0, valstep=1, color="#444444")
day_slider.label.set_color("white")
day_slider.valtext.set_color("white")

def update_slider(day_value):
    """Update map, vertical line, and TextBox when slider moves."""
    global sim_data
    day_idx = int(day_value)
    day_idx = max(0, min(sim_data["num_days"] - 1, day_idx))
    current_I = sim_data["I_hist"][day_idx].astype(float)
    gdf["infected"] = current_I
    map_collection.set_array(current_I)
    vline.set_xdata([day_idx, day_idx])
    day_box.set_val(str(day_idx))
    fig.canvas.draw_idle()

day_slider.on_changed(update_slider)

# -------------------------------------------------------------------
# 10. TextBox: type a day and jump there
# -------------------------------------------------------------------
ax_daybox = fig.add_axes([0.10, 0.08, 0.16, 0.045])
ax_daybox.set_facecolor("white")
day_box = TextBox(ax_daybox, "Select day", initial="0")
day_box.label.set_color("white")
day_box.text_disp.set_color("black")

def submit_day(text):
    try:
        val = int(text)
    except ValueError:
        return
    val = max(day_slider.valmin, min(day_slider.valmax, val))
    day_slider.set_val(val)

day_box.on_submit(submit_day)

# -------------------------------------------------------------------
# 11. Helpers to refresh plots after re-simulating
# -------------------------------------------------------------------
def update_extinction_text():
    """Update extinction text based on current extinction day in sim_data."""
    if sim_data["extinction_day"] is None:
        extinction_text.set_text("Extinction: > 1247")  # If no extinction, show > 1247 days
    else:
        extinction_text.set_text(f"Extinction day: {sim_data['extinction_day']}")  # Display extinction day

def apply_simulation_to_plots():
    """Update map, SIR curves, and color scale based on current sim_data."""
    line_S.set_data(sim_data["days"], sim_data["total_S"])
    line_I.set_data(sim_data["days"], sim_data["total_I"])
    line_R.set_data(sim_data["days"], sim_data["total_R"])
    ax_ts.set_xlim(0, max_days)
    ax_ts.set_ylim(0, N_total)
    day0_I = sim_data["I_hist"][0].astype(float)
    gdf["infected"] = day0_I
    map_collection.set_array(day0_I)
    new_max = sim_data["infected_max"]
    if new_max <= 0:
        new_max = 1
    sm.set_clim(0.0, new_max)
    map_collection.set_clim(0.0, new_max)
    cbar.update_normal(sm)
    update_extinction_text()  # Update the extinction text
    day_slider.set_val(0)
    day_box.set_val("0")
    vline.set_xdata([0, 0])
    fig.canvas.draw_idle()

# -------------------------------------------------------------------
# 12. Control buttons: <, >, Play/Pause, Randomize, Reset
# -------------------------------------------------------------------
button_color = "#333333"
hover_color = "#555555"
text_color = "white"
is_playing = False  # global flag for animation

def step_day(delta):
    current = day_slider.val
    new = current + delta
    new = max(day_slider.valmin, min(day_slider.valmax, new))
    day_slider.set_val(new)
    day_box.set_val(str(int(new)))

# "<" previous-day button
ax_prev = fig.add_axes([0.58, 0.16, 0.05, 0.04])
btn_prev = Button(ax_prev, "<", color=button_color, hovercolor=hover_color)
btn_prev.label.set_color(text_color)

def on_prev(event):
    step_day(-1)

btn_prev.on_clicked(on_prev)

# ">" next-day button
ax_next = fig.add_axes([0.64, 0.16, 0.05, 0.04])
btn_next = Button(ax_next, ">", color=button_color, hovercolor=hover_color)
btn_next.label.set_color(text_color)

def on_next(event):
    step_day(1)

btn_next.on_clicked(on_next)

# "Play/Pause" button
ax_play_pause = fig.add_axes([0.70, 0.16, 0.08, 0.04])
btn_play_pause = Button(ax_play_pause, "Play", color=button_color, hovercolor=hover_color)
btn_play_pause.label.set_color(text_color)

def on_play_pause(event):
    global is_playing
    if is_playing:
        is_playing = False
        btn_play_pause.label.set_text("Play")
    else:
        is_playing = True
        btn_play_pause.label.set_text("Pause")
    start = int(day_slider.val)
    for d in range(start, max_days + 1):
        if not is_playing:
            break
        day_slider.set_val(d)
        day_box.set_val(str(d))
        plt.pause(0.05)

btn_play_pause.on_clicked(on_play_pause)

# "Randomize" button: new simulation with same seed cells
ax_rand = fig.add_axes([0.58, 0.08, 0.20, 0.045])
btn_rand = Button(ax_rand, "Randomize", color=button_color, hovercolor=hover_color)
btn_rand.label.set_color(text_color)

def on_randomize(event):
    """Run a new stochastic simulation and refresh plots."""
    global sim_data, is_playing
    is_playing = False
    sim_data = simulate()
    apply_simulation_to_plots()

btn_rand.on_clicked(on_randomize)

# "Reset" button: back to day 0 for current simulation
ax_reset = fig.add_axes([0.80, 0.08, 0.08, 0.045])
btn_reset = Button(ax_reset, "Reset", color=button_color, hovercolor=hover_color)
btn_reset.label.set_color(text_color)

def on_reset(event):
    """Reset slider and plots to day 0 (no new simulation)."""
    global is_playing
    is_playing = False
    day_slider.set_val(0)
    day_box.set_val("0")

btn_reset.on_clicked(on_reset)

# Show the GUI
plt.show()
