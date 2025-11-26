# UAC Core Extraction

This folder contains a minimal extraction of the Universal Atrial Coordinates (UAC) fitting logic. The scripts focus on generating the Laplace boundary files required to fit the coordinate system to a left atrial geometry without depending on the full pipeline.

## CLI usage

```
# Ensure the lightweight package is on your Python path
export PYTHONPATH=src

# CARP mesh directory
python -m uac_core.fit /path/to/geometry \
  --mesh-name Labelled

# Standalone XML surface mesh (example)
python -m uac_core.fit CS/eg_atrialSurface_geo.xml

# Optionally provide your own PV landmark seeds (VTK legacy PolyData or CSV)
python -m uac_core.fit CS/eg_atrialSurface_geo.xml \
  --seed-file CS/seedsfileOUT_Landmarks.vtk
```

The command accepts either CARP-format `*.pts`/`*.elem` pairs or a standalone DOLFIN/FEniCS-style XML surface mesh and will write `PAbc1.vtx` and `PAbc2.vtx` alongside the inputs by default. Labels can be overridden with the optional arguments if your meshes use different region IDs. If you do not supply a `--seed-file`, the fitter will automatically derive pulmonary vein landmarks from the mesh geometry (centroids of the LSPV/RSPV rims).

> On Windows, use `set PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`) before running `python -m uac_core.fit`. Do **not** prefix the module path with `./src/` when using `-m`; the package name is simply `uac_core`.
