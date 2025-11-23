import geopandas as gpd

# Läs in GeoPackage
gdf = gpd.read_file("population_1km_2024.gpkg")

# Se alla kolumner
print("Kolumner i GeoPackage:")
print(gdf.columns)

# Se de första raderna för att förstå datan
print(gdf.head())

# Se datatyper
print(gdf.dtypes)

# Om du vill veta vilka kolumner som är numeriska
numeric_cols = gdf.select_dtypes(include='number').columns
print("Numeriska kolumner:")
print(numeric_cols)
