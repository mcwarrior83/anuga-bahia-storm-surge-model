... # make sure you are in the correct projection and EPSG
... # mess around with the base resolution
... # look into create_domain_from_dem()
... # set an observation point for  the SPI Brazos Santiago gage station
... # check if i can input time series data for boundary conditions
... # consider setting stage equal to port isabel gage station data

import anuga
import numpy as np
import pandas as pd
import os
from pathlib import Path
import geopandas as gpd
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow inline jshtml animations
from matplotlib import rc
rc('animation', html='jshtml')


ANUGA_CORE = Path(__file__).resolve().parent.parent.parent
data_dir = str(os.path.join(ANUGA_CORE, 'ANUGA', 'Bahia', 'input_files'))
print(f"ANUGA core directory: {data_dir}")

ARC_CORE = Path(__file__).resolve().parent.parent.parent.parent
arc_directory = str(ARC_CORE)
print(f"ARC core directory: {arc_directory}")

bahia_bsc_path = os.path.join(ARC_CORE, 'GISProjects', 'BahiaPolygon', 'BahiaBSC_Polygon.shp')
bahia_grande_path = os.path.join(data_dir, 'bahiagrande.shp')
bahiabsc_asc = Path(__file__).resolve().parent / "bahiabsc.asc"


#-------------------------- Reading Shapefile---------------------------------
shapefiles = [bahia_bsc_path, bahia_grande_path]
coord_shapefiles = []
polygon_names = ["Bahia BSC", "Bahia Grande"]

for shapefile, name in zip(shapefiles, polygon_names):
    gdf = gpd.read_file(shapefile).to_crs("EPSG:26914")

    exterior_coordinates = []
    segment_count = 0

    for polygon in gdf.geometry:
        if polygon.geom_type == "Polygon":
            polygons = [polygon]
        elif polygon.geom_type == "MultiPolygon":
            polygons = list(polygon.geoms)
        else:
            continue

        for part in polygons:
            coordinates = list(part.exterior.coords)
            exterior_coordinates.extend(coordinates)
            segment_count += max(0, len(coordinates))

    coord_shapefiles.append(exterior_coordinates)


# -------------------------- Setup Domain --------------------------------
# Polygon showing the whole area of interest
bahia_bsc_polygon = anuga.read_polygon(anuga.join(data_dir,'extent_UTM14N.csv')) # Note: must be saved as a CSV file.

# Polygon showing the particular area of interest (Bahia Grande)
#bahia_polygon = anuga.read_polygon(anuga.join(data_dir, )) # Note: must be saved as a CSV file.)
bahia_polygon = coord_shapefiles[1]

# Bathymetry Data
dem_file = anuga.join(data_dir,'bahiabsc.asc')

# Area of the whole domain
base_resolution = 20000 # m^2 | will reduce for the final model

# Area of the Bahia Grande polygon
bahia_resolution = 10000 # m^2 # will reduce for the final model

interior_regions = [[bahia_polygon, bahia_resolution]]


boundary_tags = {'inlet': [0],
                 'reflective': list(range(1, len(bahia_bsc_polygon)))
}

domain = anuga.create_domain_from_regions(
    bahia_bsc_polygon,
    boundary_tags,
    maximum_triangle_area=base_resolution,
    interior_regions=interior_regions,
)

cfl = domain.get_cfl()
print(f"CFL Number: {cfl}")

domain.set_name('bahiasww') # Name of sww file
domain.set_epsg(26914) # NAD83 / UTM zone 14N
domain.set_plotter(absolute=True, plot_dir='_plot_georeferenced')

domain.triplot(linewidth = 0.4) # remove comment if you want to see mesh layer
plt.show()

print("Number of polygon coordinates:", len(bahia_polygon))
print("Polygon CRS:", gdf.crs.to_string())
print("Polygon bounds:", gdf.total_bounds)


#-------------------- Setup Initial Conditions -------------------------------------
domain.set_quantity('elevation', filename=dem_file, location='centroids') # Use function for elevation
domain.set_quantity('friction', 0.01, location='centroids')                        # Constant friction
domain.set_quantity('stage', 0.1 , location='centroids')       # Dry Bed,  expression='elevation'

elevation = domain.get_quantity('elevation').centroid_values

fig, ax, im = domain.tripcolor(facecolors=elevation, cmap='Greys_r')


#--------------------- Geo Referencing ----------------------------------------------

ax.set_xlabel('Easting (m, EPSG:26914)')
ax.set_ylabel('Northing (m, EPSG:26914)')
plt.colorbar(im, ax=ax)
plt.title("Elevation")
plt.tight_layout()
#plt.savefig(ANUGA_DIR / 'bahia_elevation.png', dpi=200) # uncomment to save figure
plt.show() # remove comment if you want to see elevation graph


#--------------------------- Setup Boundary Conditions ------------------------------
df = pd.read_csv(os.path.join(data_dir, "spi_wl_hurricane_alex.csv"))
# Water levels must use the same vertical datum and units as the model elevation.
water_level_column = "verified_m"

df["datetime_gmt"] = pd.to_datetime(
    df["date"].astype(str).str.strip()
    + " "
    + df["time_gmt"].astype(str).str.strip(),
    format="%m/%d/%Y %H:%M",
    utc=True,
    errors="coerce"
)
df[water_level_column] = pd.to_numeric(df[water_level_column], errors="coerce")

invalid_rows = df["datetime_gmt"].isna() | df[water_level_column].isna()
if invalid_rows.any():
    print(
        f"Skipping {int(invalid_rows.sum())} rows with invalid date/time "
        "or water level."
    )
df = df.loc[~invalid_rows].sort_values("datetime_gmt")
df = df.drop_duplicates(subset="datetime_gmt", keep="last").reset_index(drop=True)

if len(df) < 2:
    raise ValueError("At least two valid tide observations are required.")

df["elapsed_seconds"] = (
    df["datetime_gmt"] - df["datetime_gmt"].iloc[0]
).dt.total_seconds()

water_level_times = df["elapsed_seconds"].to_numpy(dtype=float)
water_level_values = df[water_level_column].to_numpy(dtype=float)
simulation_duration = float(water_level_times[-1])

Br = anuga.Reflective_boundary(domain)

def tide_stage(t):
    return float(np.interp(t, water_level_times, water_level_values))

Bt = anuga.Time_boundary(
    domain,
    function=lambda t: [tide_stage(t), 0.0, 0.0] # might need to change to reflect documentation
)
domain.set_boundary({'inlet': Bt,
                     'reflective': Br})


"""
# ----------------------------- Setup Inflow ----------------------------------------
center = (685798.579000, 2884323.121900) # selected southern portion of channel 
radius = 50 # metres; must include a mesh centroid (nearest is 34.3 m away) # this is wrong
region0 = anuga.Region(domain, center=center, radius=radius)
fixed_inflow = anuga.Inlet_operator(domain, region0, Q=30.0) # need to find out mean flow rate
"""


# ------------------------------- Evolution -------------------------------------------
for t in domain.evolve(yieldstep=7200, duration=simulation_duration):
    domain.save_depth_frame(vmin=0.0, vmax=1.0)
    domain.print_timestepping_statistics()

domain.make_depth_animation()
