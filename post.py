import netCDF4
import numpy as np

with netCDF4.Dataset("bahiasww.sww") as ds:
    time = np.asarray(ds.variables["time"][:])
    stage = np.asarray(ds.variables["stage_c"][:])
    bed = np.asarray(ds.variables["elevation_c"][:])
    depth = stage - bed

    print("EPSG:", ds.epsg)
    print("Time range (s):", time[0], time[-1])
    print("Time strictly increasing:", np.all(np.diff(time) > 0))
    print("Stage finite:", np.isfinite(stage).all())
    print("Bed elevations finite:", np.isfinite(bed).all())
    print("Depth finite:", np.isfinite(depth).all())
    print("NaN bed values:", np.isnan(bed).sum(), "of", bed.size)
    