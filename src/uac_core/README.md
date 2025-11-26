# UAC Core Extraction

This folder contains a minimal extraction of the Universal Atrial Coordinates (UAC) fitting logic. The scripts focus on generating the Laplace boundary files required to fit the coordinate system to a left atrial geometry without depending on the full pipeline.

## CLI usage

```
# CARP mesh directory
python -m uac_core.fit /path/to/geometry \
  --mesh-name Labelled \
  --seed-file seedsfileOUT_Landmarks.vtk

# Standalone XML surface mesh
python -m uac_core.fit /path/to/eg_atrialSurface_geo.xml \
  --seed-file /path/to/seedsfileOUT_Landmarks.vtk
```

The command accepts either CARP-format `*.pts`/`*.elem` pairs or a standalone XML surface mesh (VTK XML or legacy DOLFIN format) and will write `PAbc1.vtx` and `PAbc2.vtx` alongside the inputs by default. Labels can be overridden with the optional arguments if your meshes use different region IDs.
