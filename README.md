# Closed-Conduit Cave Hydraulics Model

This repository contains the Python code, input data, and primary outputs used to simulate flood flow in a closed-conduit cave system (Model Cave, Great Basin National Park, NV).

The model compares three hydraulic equations:
- Darcy–Weisbach
- Manning’s equation
- Hazen–Williams equation

## Repository structure

- `src/` — Python model code
- `inputs/` — required input files
- `outputs/` — primary model results

## Input files

- `survey.txt` — cave survey data with LRUD measurements
- `Flow data ell.csv` — scallop-derived velocities and hydraulic parameters
- `Survey_and_scallop_data.csv` — scallop data used to estimate discharge

## Output files

- Model results exported as CSV files, including:
  - `*_nodes.csv` — node-by-node hydraulic results (HGL, EGL, velocity, Reynolds number, friction slope)
  - `geometry_nodes.csv` — conduit geometry along the passage (area, hydraulic radius, elevation, coordinates)
  - `method_summary_metrics.csv` — summary statistics for each method
  - `method_comparison_metrics.csv` — comparison between hydraulic equations
  - `validation_metrics_summary.csv` — model validation metrics
  - `validation_point_comparison.csv` — observed vs modeled values at validation sites
  - `darcy_uncertainty_envelope.csv` — Monte Carlo uncertainty bounds for the primary (Darcy–Weisbach) model

## Notes

- This is a one-dimensional steady-state hydraulic model
- Geometry is approximated as elliptical using LRUD survey data
- Flow is based on scallop-derived velocities

## Author

Ripley Taylor  
M.S. Geological Sciences — Ohio University
