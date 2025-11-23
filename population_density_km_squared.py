import geopandas as gpd
import matplotlib.pyplot as plt

# Läs in GPkg
gdf = gpd.read_file("population_1km_2024.gpkg")

# Använd 'beftotalt' som total befolkning per cell
population_col = 'beftotalt'
gdf['total'] = gdf[population_col].astype(float)

# Plot
gdf.plot(
    column='total',
    cmap='OrRd',
    legend=True,
    edgecolor='black',
    linewidth=0.05,
    figsize=(10, 12)
)

plt.show()
