gpkg_path = "population_1km_2024.gpkg"  # Path to your GeoPackage file
population_col = "beftotalt"           # Name of the population column in the GPKG

import geopandas as gpd                 # GeoPandas for spatial data
import matplotlib.pyplot as plt         # Matplotlib for plotting
from matplotlib.widgets import Slider, Button, TextBox  # Widgets for interactivity
from matplotlib.colors import LinearSegmentedColormap   # For custom colormaps
from matplotlib.gridspec import GridSpec                # For layout control
import numpy as np                      # NumPy for numerical operations

# ============================================================
# 1. Load data and prepare grid
# ============================================================

# We do NOT set a random seed so each Randomize run is different
# np.random.seed(42)  # You can uncomment to make runs reproducible

gdf = gpd.read_file(gpkg_path)  # Read the GeoPackage as a GeoDataFrame

# Make sure the population column is float and replace NaNs with 0
gdf[population_col] = gdf[population_col].astype(float).fillna(0.0)

# Store the population as "total" per cell
gdf["total"] = gdf[population_col].astype(float)

# Extract total population as a NumPy array
total_pop = gdf["total"].values

# Round to integers because we want integer counts of people
N = np.round(total_pop).astype(int)

# Number of spatial cells
num_cells = len(gdf)

# Ensure population is not negative
N = np.clip(N, 0, None)

# Total population over all cells, used for scaling
N_total = int(N.sum())

# If total population is zero, the model cannot run
if N_total == 0:
    raise ValueError("Total population is zero; cannot simulate epidemic.")

# Avoid division by zero when computing I/N by creating a "safe" N
N_safe = np.where(N > 0, N, 1)

# ============================================================
# 2. Seed cells (fixed starting locations)
#    - Cells around ~12 500 inhabitants
#    - Less than 5% of non-zero cells are chosen
# ============================================================

# Boolean mask of cells that have non-zero population
nonzero_mask = N > 0

# Indices of cells with non-zero population
nonzero_indices = np.where(nonzero_mask)[0]

# If all cells have zero population, abort
if len(nonzero_indices) == 0:
    raise ValueError("All cells have zero population; cannot simulate epidemic.")

# Choose about 4% of non-empty cells as initial infection seed cells
num_seed_cells = max(1, int(0.04 * len(nonzero_indices)))

# Target population size for seed cells (~12 500)
target_pop = 12500.0

# Population values only for non-zero cells
pop_nonzero = total_pop[nonzero_mask]

# Distance from each cell's population to the target population
distance_to_target = np.abs(pop_nonzero - target_pop)

# Sort indices by how close they are to the target population
sorted_idx = np.argsort(distance_to_target)

# Map sorted indices back to original cell indices
sorted_nonzero_indices = nonzero_indices[sorted_idx]

# Select the closest cells as seed cells
seed_indices = sorted_nonzero_indices[:num_seed_cells]  # These are fixed across runs

# ============================================================
# 3. Extended SIR parameters and mixing
#    - beta_local, gamma: infection and recovery
#    - mu: mortality rate (I -> D)
#    - alpha: waning immunity rate (R -> S)
#    - Lockdown: reduces contact rates in a time window
# ============================================================

infectious_period = 7.0             # Average infectious period in days
gamma = 1.0 / infectious_period     # Recovery probability per day

R0_local = 2.5                      # Basic reproduction number for local mixing
beta_local = R0_local * gamma       # Local infection rate per day

beta_global = 0.05 * beta_local     # Small global infection component (between cells)

mu = 0.001                          # Mortality probability per day for infected agents

waning_immunity_days = 365.0        # Average days of immunity
alpha = 1.0 / waning_immunity_days  # Probability per day that R becomes S

# Lockdown parameters (inspired by the textbook exercise)
use_lockdown = True                 # Whether lockdown is active in the model
lockdown_start_day = 60             # Day at which lockdown starts
lockdown_duration = 200             # Duration of lockdown in days
lockdown_contact_factor = 0.1       # Contacts during lockdown are 10% of normal

# Normalize population by maximum to get values between 0 and 1
if N.max() > 0:
    pop_norm = N / N.max()
else:
    pop_norm = np.zeros_like(N, dtype=float)

# Weight for global mixing: high-pop cells get stronger global influence
pop_weight_for_global = np.sqrt(pop_norm)

max_days = 1247                      # Total number of days to simulate (0..max_days)

# ============================================================
# 4. Simulation function (one stochastic realization)
#    Compartments: S (susceptible), I (infected), R (recovered), D (dead)
#    - Day 0: no infections
#    - Day 1: infections are seeded in high-pop cells
#    - Day 2..max_days: SIRD + waning immunity + optional lockdown
# ============================================================

def simulate():
    # -----------------------------
    # Day 0: all susceptible, no infection, no recovered, no dead
    # -----------------------------
    S0 = N.copy()                             # Everyone starts susceptible
    I0 = np.zeros_like(N, dtype=int)         # No infected initially
    R0 = np.zeros_like(N, dtype=int)         # No recovered initially
    D0 = np.zeros_like(N, dtype=int)         # No dead initially

    # Lists to store history of each compartment over time
    S_hist = [S0.copy()]                     # Add day 0 S
    I_hist = [I0.copy()]                     # Add day 0 I
    R_hist = [R0.copy()]                     # Add day 0 R
    D_hist = [D0.copy()]                     # Add day 0 D

    # -----------------------------
    # Day 1: seed infections in the chosen high-population cells
    # -----------------------------
    S = S0.copy()                            # Start from day 0 state
    I = I0.copy()
    R = R0.copy()
    D = D0.copy()

    # Random infected fraction (1–5%) in each seed cell
    initial_infected_fraction = np.random.uniform(0.01, 0.05, size=num_seed_cells)

    # Loop over each seed cell and infect some fraction of its population
    for idx, frac in zip(seed_indices, initial_infected_fraction):
        new_inf = int(round(frac * N[idx]))  # Number of newly infected
        new_inf = max(new_inf, 1)            # At least 1 infected if possible
        new_inf = min(new_inf, S[idx])       # Cannot infect more than S
        S[idx] -= new_inf                    # Reduce susceptible
        I[idx] += new_inf                    # Increase infected

    # Save state at day 1
    S_hist.append(S.copy())
    I_hist.append(I.copy())
    R_hist.append(R.copy())
    D_hist.append(D.copy())

    # -----------------------------
    # Days 2..max_days: simulate dynamics each day
    # -----------------------------
    for day in range(2, max_days + 1):
        # Default contact factor (no lockdown)
        contact_factor = 1.0

        # If lockdown is active and we are in the lockdown window, reduce contacts
        if use_lockdown:
            if lockdown_start_day <= day < lockdown_start_day + lockdown_duration:
                contact_factor = lockdown_contact_factor

        # Adjust local and global infection rates based on lockdown
        beta_local_eff = beta_local * contact_factor
        beta_global_eff = beta_global * contact_factor

        # Total infected in the whole population
        I_total = I.sum()

        # Local infected fraction (I/N) for each cell
        prevalence_local = I / N_safe

        # Local infection intensity for each cell
        lambda_local = beta_local_eff * prevalence_local

        # Probability of infection from local interactions
        p_local = 1.0 - np.exp(-lambda_local)

        # Global infected fraction (all infected / all population)
        prevalence_global = I_total / N_total if N_total > 0 else 0.0

        # Base global infection intensity
        lambda_global_base = beta_global_eff * prevalence_global

        # Cell-specific global intensity, scaled by population weight
        lambda_global = lambda_global_base * pop_weight_for_global

        # Probability of infection from global interactions
        p_global = 1.0 - np.exp(-lambda_global)

        # Total infection probability per cell
        p_inf = p_local + p_global

        # Clip probabilities to [0, 1]
        p_inf = np.clip(p_inf, 0.0, 1.0)

        # Draw new infections from susceptible using a binomial per cell
        new_infections = np.random.binomial(S, p_inf)

        # Draw recoveries from infected using gamma
        new_recoveries = np.random.binomial(I, gamma)

        # Subtract recovered from infected
        I_after_rec = I - new_recoveries

        # Draw deaths from remaining infected using mu
        new_deaths = np.random.binomial(I_after_rec, mu)

        # Subtract deaths from infected
        I_after_rec_death = I_after_rec - new_deaths

        # Waning immunity: some recovered become susceptible again
        new_S_from_R = np.random.binomial(R, alpha)

        # Update S: lose new infections, gain those losing immunity
        S = S - new_infections + new_S_from_R

        # Update I: gain new infections, lose recovered and dead
        I = I_after_rec_death + new_infections

        # Update R: gain recovered, lose those losing immunity
        R = R + new_recoveries - new_S_from_R

        # Update D: gain new deaths
        D = D + new_deaths

        # Save current day state
        S_hist.append(S.copy())
        I_hist.append(I.copy())
        R_hist.append(R.copy())
        D_hist.append(D.copy())

    # Convert histories to NumPy arrays for easier handling
    S_hist = np.array(S_hist)
    I_hist = np.array(I_hist)
    R_hist = np.array(R_hist)
    D_hist = np.array(D_hist)

    # Number of simulated days (should be max_days + 1)
    num_days = S_hist.shape[0]

    # Day indices: 0, 1, ..., max_days
    days = np.arange(num_days)

    # Total S, I, R, D at each day (sum over all cells)
    total_S = S_hist.sum(axis=1)
    total_I = I_hist.sum(axis=1)
    total_R = R_hist.sum(axis=1)
    total_D = D_hist.sum(axis=1)

    # Make sure total infected at day 0 is exactly 0
    total_I[0] = 0

    # Find extinction day: first day after day 0 when total_I becomes 0 again
    extinction_day = None
    for d in range(1, num_days):
        if total_I[d] == 0:
            extinction_day = d
            break

    # For the color scale: use the 99th percentile of infected counts
    positive_vals = I_hist[I_hist > 0]  # All positive infected cell counts

    if positive_vals.size > 0:
        infected_max = int(np.ceil(np.percentile(positive_vals, 99)))
    else:
        infected_max = 1

    if infected_max < 1:
        infected_max = 1

    # Return all data in a dictionary
    return {
        "S_hist": S_hist,
        "I_hist": I_hist,
        "R_hist": R_hist,
        "D_hist": D_hist,
        "total_S": total_S,
        "total_I": total_I,
        "total_R": total_R,
        "total_D": total_D,
        "days": days,
        "num_days": num_days,
        "infected_max": infected_max,
        "extinction_day": extinction_day,
    }

# Run the first simulation to initialize everything
sim_data = simulate()

# ============================================================
# 5. Prepare initial map (infected count, day 0 = all 0)
# ============================================================

# At day 0, there are no infections, so this should be all zeros
gdf["infected"] = sim_data["I_hist"][0].astype(float)

# ============================================================
# 6. Figure layout: more spacing using GridSpec
# ============================================================

fig = plt.figure(figsize=(15, 6))                           # Create a figure of size 15x6 inches

# Use GridSpec to split figure into 1 row, 2 columns with different widths
gs = GridSpec(1, 2, width_ratios=[1.1, 1.4], wspace=0.3, figure=fig)

# Left subplot for the map
ax_map = fig.add_subplot(gs[0, 0])

# Right subplot for the time series
ax_ts = fig.add_subplot(gs[0, 1])

# Leave extra space at the bottom for sliders and buttons
plt.subplots_adjust(bottom=0.26)

# Dark gray as overall background for the figure
dark_gray = "#222222"
fig.patch.set_facecolor(dark_gray)

# Map background same as figure background
ax_map.set_facecolor(dark_gray)

# Time-series background slightly lighter gray so it stands out
lighter_gray = "#333333"
ax_ts.set_facecolor(lighter_gray)

# ============================================================
# 7. Map: black → yellow → red gradient + colorbar
# ============================================================

# Define a custom colormap from black to yellow to red
cmap = LinearSegmentedColormap.from_list(
    "black_yellow_red",
    ["#000000", "#ffff00", "#ff0000"]
)

# Maximum infected value used for color scaling
infected_max = sim_data["infected_max"]

# Plot the GeoDataFrame colored by "infected"
gdf.plot(
    column="infected",       # Column used for colors
    ax=ax_map,               # Axis to draw on
    cmap=cmap,               # Our custom colormap
    vmin=0.0,                # Minimum value for colormap
    vmax=infected_max,       # Maximum value for colormap
    linewidth=0,             # No borders around polygons
    edgecolor="none"         # No edge color
)

# Set a short title for the map
ax_map.set_title("Infected map", color="white")

# Remove axis (ticks, labels) for a clean map
ax_map.set_axis_off()

# Matplotlib stores the polygon collection here; we use it to update colors later
map_collection = ax_map.collections[0]

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(vmin=0.0, vmax=infected_max)
)
sm._A = []  # Dummy array to make ScalarMappable work

# Add a colorbar to show how color maps to infected count
cbar = fig.colorbar(sm, ax=ax_map, fraction=0.05, pad=0.02)
cbar.set_label("Infected count", color="white")  # Label for colorbar
cbar.outline.set_edgecolor("white")             # Make colorbar border white
plt.setp(cbar.ax.get_yticklabels(), color="white")  # Color of tick labels

# ============================================================
# 8. Agent-based SIR plot (S, I, R, D) + lockdown shading
# ============================================================

# Plot total S over time
line_S, = ax_ts.plot(
    sim_data["days"],
    sim_data["total_S"],
    label="S",
    color="#1f77b4"
)

# Plot total I over time
line_I, = ax_ts.plot(
    sim_data["days"],
    sim_data["total_I"],
    label="I",
    color="#ff7f0e"
)

# Plot total R over time
line_R, = ax_ts.plot(
    sim_data["days"],
    sim_data["total_R"],
    label="R",
    color="#2ca02c"
)

# Plot total D (dead) over time
line_D, = ax_ts.plot(
    sim_data["days"],
    sim_data["total_D"],
    label="D",
    color="#aa00ff"
)

# Label the x-axis as "Day"
ax_ts.set_xlabel("Day", color="white")

# Label the y-axis as "Number of individuals"
ax_ts.set_ylabel("Number of individuals", color="white")

# Set the title as requested
ax_ts.set_title("Agent-based SIR model", color="white")

# Fix y-axis range from 0 to total population
ax_ts.set_ylim(0, N_total)

# Fix x-axis from day 0 to max_days
ax_ts.set_xlim(0, max_days)

# Make x and y tick labels white
ax_ts.tick_params(colors="white")

# Make plot spines (borders) white
for spine in ax_ts.spines.values():
    spine.set_color("white")

# Add a faint grid
ax_ts.grid(color="gray", alpha=0.3)

# If lockdown is used, shade the lockdown period in the time series
if use_lockdown:
    ax_ts.axvspan(
        lockdown_start_day,
        lockdown_start_day + lockdown_duration,
        color="gray",
        alpha=0.15
    )

# Add legend and style it
legend = ax_ts.legend(facecolor="#333333", edgecolor="none")
for text in legend.get_texts():
    text.set_color("white")

# Add a vertical line that shows the currently selected day
vline = ax_ts.axvline(0, linestyle="--", color="white", alpha=0.7)

# Function to format the extinction text
def format_extinction(ext_day):
    if ext_day is None:
        return "Extinction: > 1247"
    else:
        return f"Extinction day: {ext_day}"

# Add text to show the extinction day in the top-left corner of the SIR plot
extinction_text = ax_ts.text(
    0.02, 0.95, format_extinction(sim_data["extinction_day"]),
    transform=ax_ts.transAxes,
    color="white",
    fontsize=9,
    va="top"
)

# ============================================================
# 9. Day slider (to scrub through simulation days)
# ============================================================

# Create an axis area for the slider [left, bottom, width, height]
ax_slider = fig.add_axes([0.10, 0.16, 0.45, 0.03])

# Create the Slider widget
day_slider = Slider(
    ax=ax_slider,          # Axis to draw slider on
    label="Day",           # Label shown next to the slider
    valmin=0,              # Minimum slider value
    valmax=max_days,       # Maximum slider value
    valinit=0,             # Initial slider value
    valstep=1,             # Step size (one day)
    color="#444444"        # Color of the slider bar
)

# Make the slider label white
day_slider.label.set_color("white")

# Make the slider current value text white
day_slider.valtext.set_color("white")

# Callback function called whenever the slider value changes
def update_slider(day_value):
    """Update map, vertical line, and TextBox when slider moves."""
    global sim_data

    # Convert slider value to integer day index
    day_idx = int(day_value)

    # Clamp day index to valid range
    day_idx = max(0, min(sim_data["num_days"] - 1, day_idx))

    # Get infected counts for this day
    current_I = sim_data["I_hist"][day_idx].astype(float)

    # Update the GeoDataFrame column
    gdf["infected"] = current_I

    # Update the color values of the map
    map_collection.set_array(current_I)

    # Move the vertical line in the SIR plot
    vline.set_xdata([day_idx, day_idx])

    # Update the TextBox to show the current day
    day_box.set_val(str(day_idx))

    # Redraw the figure
    fig.canvas.draw_idle()

# Connect the slider to the callback
day_slider.on_changed(update_slider)

# ============================================================
# 10. TextBox: type a day and jump there (formal + visible)
# ============================================================

# Create an axis for the text box
ax_daybox = fig.add_axes([0.10, 0.08, 0.16, 0.045])

# Set background of the text box area to white so text is visible
ax_daybox.set_facecolor("white")

# Create TextBox widget with label "Select day"
day_box = TextBox(ax_daybox, "Select day", initial="0")

# Make label black (so it shows on white background)
day_box.label.set_color("white")

# Make the typed text black
day_box.text_disp.set_color("black")

# Callback for when user presses Enter in the text box
def submit_day(text):
    try:
        val = int(text)  # Try to convert text to integer
    except ValueError:
        return           # If conversion fails, do nothing
    # Clamp the chosen day between min and max of slider
    val = max(day_slider.valmin, min(day_slider.valmax, val))
    # Update the slider, which also updates the plots
    day_slider.set_val(val)

# Connect the text box to the submit callback
day_box.on_submit(submit_day)

# ============================================================
# 11. Helper: apply a new simulation to the plots
# ============================================================

# Helper to update the extinction text when sim_data changes
def update_extinction_text():
    extinction_text.set_text(format_extinction(sim_data["extinction_day"]))

# Helper to refresh everything after we re-simulate
def apply_simulation_to_plots():
    """Update map, SIR curves and color scale based on sim_data."""
    # Update S, I, R, D lines with new data
    line_S.set_data(sim_data["days"], sim_data["total_S"])
    line_I.set_data(sim_data["days"], sim_data["total_I"])
    line_R.set_data(sim_data["days"], sim_data["total_R"])
    line_D.set_data(sim_data["days"], sim_data["total_D"])

    # Keep x and y limits consistent
    ax_ts.set_xlim(0, max_days)
    ax_ts.set_ylim(0, N_total)

    # Update map for day 0 (no infection)
    day0_I = sim_data["I_hist"][0].astype(float)
    gdf["infected"] = day0_I
    map_collection.set_array(day0_I)

    # Update color scale for the map based on new simulation
    new_max = sim_data["infected_max"]
    if new_max <= 0:
        new_max = 1
    sm.set_clim(0.0, new_max)
    map_collection.set_clim(0.0, new_max)
    cbar.update_normal(sm)

    # Update extinction day text
    update_extinction_text()

    # Reset slider to day 0
    day_slider.set_val(0)

    # Reset text box to day 0
    day_box.set_val("0")

    # Reset vertical line to day 0
    vline.set_xdata([0, 0])

    # Redraw everything
    fig.canvas.draw_idle()

# ============================================================
# 12. Buttons: <, >, Play/Pause, Randomize, Reset
# ============================================================

# Color of buttons
button_color = "#333333"

# Hover color of buttons
hover_color = "#555555"

# Text color on buttons
text_color = "white"

# Global flag to know if animation is playing
is_playing = False

# Helper to move one day forward or backward
def step_day(delta):
    current = day_slider.val                 # Current slider value
    new = current + delta                    # Add delta (+1 or -1)
    new = max(day_slider.valmin, min(day_slider.valmax, new))  # Clamp
    day_slider.set_val(new)                  # Update slider (and plots)
    day_box.set_val(str(int(new)))           # Update text box

# "<" button: previous day
ax_prev = fig.add_axes([0.58, 0.16, 0.05, 0.04])
btn_prev = Button(ax_prev, "<", color=button_color, hovercolor=hover_color)
btn_prev.label.set_color(text_color)

def on_prev(event):
    step_day(-1)

btn_prev.on_clicked(on_prev)

# ">" button: next day
ax_next = fig.add_axes([0.64, 0.16, 0.05, 0.04])
btn_next = Button(ax_next, ">", color=button_color, hovercolor=hover_color)
btn_next.label.set_color(text_color)

def on_next(event):
    step_day(1)

btn_next.on_clicked(on_next)

# "Play/Pause" button: toggle play/pause
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

    start = int(day_slider.val)  # Start from the current slider value
    for d in range(start, max_days + 1):
        if not is_playing:       # If paused, break out of loop
            break
        day_slider.set_val(d)    # Move slider to day d
        day_box.set_val(str(d))  # Update text box
        plt.pause(0.05)          # Wait a bit to show animation

btn_play_pause.on_clicked(on_play_pause)

# "Randomize" button: new simulation (same seed locations, new randomness)
ax_rand = fig.add_axes([0.58, 0.08, 0.20, 0.045])
btn_rand = Button(ax_rand, "Randomize", color=button_color, hovercolor=hover_color)
btn_rand.label.set_color(text_color)

def on_randomize(event):
    """New random simulation; update all plots."""
    global sim_data, is_playing
    is_playing = False          # Ensure we are not in playing mode
    sim_data = simulate()       # Run a new simulation realization
    apply_simulation_to_plots() # Refresh plots with new data

btn_rand.on_clicked(on_randomize)

# "Reset" button: go back to day 0 for current simulation
ax_reset = fig.add_axes([0.80, 0.08, 0.08, 0.045])
btn_reset = Button(ax_reset, "Reset", color=button_color, hovercolor=hover_color)
btn_reset.label.set_color(text_color)

def on_reset(event):
    """Reset to day 0 for the current simulation."""
    global is_playing
    is_playing = False          # Ensure play is stopped
    day_slider.set_val(0)       # Move slider to day 0
    day_box.set_val("0")        # Update text box to 0

btn_reset.on_clicked(on_reset)

# Finally, show the interactive plot window
plt.show()
