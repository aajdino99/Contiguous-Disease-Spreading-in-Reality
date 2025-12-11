# -*- coding: utf-8 -*- 
""" Spatial SIR model for COVID‑19 on a 1 km grid over Sweden. 
- Each cell of the GeoPackage 'population_1km_2024.gpkg' is one metapopulation. 
- Compartments per cell: S, I, R (with waning immunity R -> S). 
- DYNAMIC LOCKDOWN with STABILITY:
    - Triggers based on infection % thresholds.
    - Enforces MINIMUM DURATION and COOLDOWN periods to prevent rapid toggling.
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
# np.random.seed(42) 

# Read the 1 km population grid for Sweden
gdf = gpd.read_file(gpkg_path)
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
infectious_period = 7.0  
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
lockdown_trigger_start = 5.0  # Start if > % infected
lockdown_trigger_end = 2.0    # End if < % infected

# 2. Stability Constraints (prevent rapid switching)
min_lockdown_duration = 21    # Once started, must last at least 3 weeks
min_cooldown_duration = 14    # Once ended, cannot restart for 2 weeks

lockdown_contact_factor = 0.2  # % reduction

# --- Global mixing weights ------------------------------------------
if N.max() > 0:
    pop_norm = N / N.max()
else:
    pop_norm = np.zeros_like(N, dtype=float)
pop_weight_for_global = np.sqrt(pop_norm) 

max_days = 1247

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
                # Rule 1: Must exceed minimum duration
                # Rule 2: Infection must drop below end threshold
                if days_in_current_state >= min_lockdown_duration:
                    if pct_infected < lockdown_trigger_end:
                        lockdown_active = False
                        last_state_change_day = day
            else:
                # CURRENTLY OPEN
                # Rule 1: Must exceed cooldown duration
                # Rule 2: Infection must rise above start threshold
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
            new_infections = np.floor(new_infections * 0.1)

        new_recoveries = np.random.binomial(I.astype(int), gamma)

        S = S - new_infections
        I = I + new_infections - new_recoveries
        R = R + new_recoveries

        S_hist.append(S.copy())
        I_hist.append(I.copy())
        R_hist.append(R.copy())

    # --- Post-Processing ---
    S_hist = np.array(S_hist)
    I_hist = np.array(I_hist)
    R_hist = np.array(R_hist)
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
        "lockdown_intervals": lockdown_intervals
    }

sim_data = simulate()

# -------------------------------------------------------------------
# 5. Prepare initial map
# -------------------------------------------------------------------
gdf["infected"] = sim_data["I_hist"][0].astype(float)

# -------------------------------------------------------------------
# 6. Figure layout
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
# 7. Map
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
cbar.ax.set_visible(True)  # show/hide the colorbar
cbar.set_label("Infected count", color="white")
cbar.outline.set_edgecolor("white")
plt.setp(cbar.ax.get_yticklabels(), color="white")

# -------------------------------------------------------------------
# 8. SIR time series plot + Lockdown Shading
# -------------------------------------------------------------------
line_S, = ax_ts.plot(sim_data["days"], sim_data["total_S"], label="S", color="#1f77b4")
line_I, = ax_ts.plot(sim_data["days"], sim_data["total_I"], label="I", color="#ff7f0e")
line_R, = ax_ts.plot(sim_data["days"], sim_data["total_R"], label="R", color="#2ca02c")
ax_ts.set_xlabel("Day", color="white")
ax_ts.set_ylabel("Number of individuals", color="white")
ax_ts.set_title("Spatial SIR - Dynamic Lockdown (Min duration 21d)", color="white")
ax_ts.set_ylim(0, N_total)
ax_ts.tick_params(colors="white")
for spine in ax_ts.spines.values(): spine.set_color("white")
ax_ts.grid(color="gray", alpha=0.3)

# Legend setup with initial values
legend = ax_ts.legend(facecolor="#333333", edgecolor="none", loc="upper right")
for text in legend.get_texts(): 
    text.set_color("white")

vline = ax_ts.axvline(0, linestyle="--", color="white", alpha=0.7)
extinction_text = ax_ts.text(0.02, 0.95, "", transform=ax_ts.transAxes, color="white", fontsize=9, va="top")
# New Lockdown text indicator
lockdown_text = ax_ts.text(0.02, 0.88, "", transform=ax_ts.transAxes, color="#ff4444", fontsize=12, fontweight="bold", va="top")

lockdown_spans = []

def draw_lockdown_spans(intervals):
    global lockdown_spans
    for span in lockdown_spans:
        span.remove()
    lockdown_spans = []
    
    for (start, end) in intervals:
        if end > start:
            # Add a slight visual gap or alpha to make distinct blocks visible
            span = ax_ts.axvspan(start, end, color="gray", alpha=0.3, label="Lockdown")
            lockdown_spans.append(span)

draw_lockdown_spans(sim_data["lockdown_intervals"])

def update_extinction_text():
    d = sim_data["extinction_day"]
    txt = "Extinction: > 1247" if d is None else f"Extinction day: {d}"
    extinction_text.set_text(txt)

update_extinction_text()
ax_ts.set_xlim(0, sim_data["extinction_day"] if sim_data["extinction_day"] else max_days)

# -------------------------------------------------------------------
# 9. Day slider
# -------------------------------------------------------------------
ax_slider = fig.add_axes([0.10, 0.16, 0.45, 0.03])
day_slider = Slider(ax=ax_slider, label="Day", valmin=0, valmax=max_days, valinit=0, valstep=1, color="#444444")
day_slider.label.set_color("white")
day_slider.valtext.set_color("white")

def update_slider(day_value):
    global sim_data
    day_idx = int(day_value)
    day_idx = max(0, min(sim_data["num_days"] - 1, day_idx))
    current_I = sim_data["I_hist"][day_idx].astype(float)
    gdf["infected"] = current_I
    map_collection.set_array(current_I)
    vline.set_xdata([day_idx, day_idx])
    day_box.set_val(str(day_idx))
    
    # Update legend text with current S, I, R values
    val_S = int(sim_data["total_S"][day_idx])
    val_I = int(sim_data["total_I"][day_idx])
    val_R = int(sim_data["total_R"][day_idx])
    
    if legend:
        texts = legend.get_texts()
        if len(texts) >= 3:
            texts[0].set_text(f"S: {val_S:,}")
            texts[1].set_text(f"I: {val_I:,}")
            texts[2].set_text(f"R: {val_R:,}")

    # Check and update lockdown status text
    is_locked = False
    for start, end in sim_data["lockdown_intervals"]:
        if start <= day_idx < end:
            is_locked = True
            break
    
    if is_locked:
        lockdown_text.set_text("LOCKDOWN")
    else:
        lockdown_text.set_text("")

    fig.canvas.draw_idle()

day_slider.on_changed(update_slider)

# -------------------------------------------------------------------
# 10. TextBox
# -------------------------------------------------------------------
ax_daybox = fig.add_axes([0.10, 0.08, 0.16, 0.045])
ax_daybox.set_facecolor("white")
day_box = TextBox(ax_daybox, "Select day", initial="0")
day_box.label.set_color("white")
day_box.text_disp.set_color("black")

def submit_day(text):
    try: val = int(text)
    except ValueError: return
    val = max(day_slider.valmin, min(day_slider.valmax, val))
    day_slider.set_val(val)

day_box.on_submit(submit_day)

# -------------------------------------------------------------------
# 11. Re-simulation helpers
# -------------------------------------------------------------------
def apply_simulation_to_plots():
    line_S.set_data(sim_data["days"], sim_data["total_S"])
    line_I.set_data(sim_data["days"], sim_data["total_I"])
    line_R.set_data(sim_data["days"], sim_data["total_R"])
    
    limit_day = sim_data["extinction_day"] if sim_data["extinction_day"] else max_days
    ax_ts.set_xlim(0, limit_day)
    ax_ts.set_ylim(0, N_total)
    
    draw_lockdown_spans(sim_data["lockdown_intervals"])
    
    day0_I = sim_data["I_hist"][0].astype(float)
    gdf["infected"] = day0_I
    map_collection.set_array(day0_I)
    
    new_max = sim_data["infected_max"]
    if new_max <= 0: new_max = 1
    sm.set_clim(0.0, new_max)
    map_collection.set_clim(0.0, new_max)
    cbar.update_normal(sm)
    
    update_extinction_text()
    day_slider.set_val(0)
    day_box.set_val("0")
    vline.set_xdata([0, 0])
    
    # Initialize legend and text for Day 0
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

ax_prev = fig.add_axes([0.58, 0.16, 0.05, 0.04])
btn_prev = Button(ax_prev, "<", color=button_color, hovercolor=hover_color)
btn_prev.label.set_color(text_color)
btn_prev.on_clicked(lambda e: step_day(-1))

ax_next = fig.add_axes([0.64, 0.16, 0.05, 0.04])
btn_next = Button(ax_next, ">", color=button_color, hovercolor=hover_color)
btn_next.label.set_color(text_color)
btn_next.on_clicked(lambda e: step_day(1))

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
        if not is_playing: break
        day_slider.set_val(d)
        day_box.set_val(str(d))
        plt.pause(0.05)

btn_play_pause.on_clicked(on_play_pause)

ax_rand = fig.add_axes([0.58, 0.08, 0.20, 0.045])
btn_rand = Button(ax_rand, "Randomize", color=button_color, hovercolor=hover_color)
btn_rand.label.set_color(text_color)
btn_rand.on_clicked(lambda e: (globals().update(is_playing=False), globals().update(sim_data=simulate()), apply_simulation_to_plots()))

ax_reset = fig.add_axes([0.80, 0.08, 0.08, 0.045])
btn_reset = Button(ax_reset, "Reset", color=button_color, hovercolor=hover_color)
btn_reset.label.set_color(text_color)
btn_reset.on_clicked(lambda e: (globals().update(is_playing=False), day_slider.set_val(0), day_box.set_val("0")))

# Initial call to set legend and text values correctly on startup
update_slider(0)

plt.show()