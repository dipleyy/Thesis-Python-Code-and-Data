# Model Cave Flood (closed-conduit epiphreatic lift passage flood) Simulation
# --------------------------------------------------------------------
# What this script does
# 1) Reads cave survey shots and builds a 3D centerline
# 2) Converts LRUD measurements into elliptical hydraulic geometry
#    following Springer (2004): A = pi*a*b, Rh = A/P (Ramanujan perimeter)
# 3) Computes a scallop-consistent baseline discharge Q0 = median(V * A)
# 4) Estimates Darcy-Weisbach friction factor from scallops using the
#    Blumberg and Curl (1974)  equation
# 5) Runs steady 1D hydraulic calculations for Darcy-Weisbach, Manning, and Hazen-Williams
# 7) Adds diagnostics files:
# dimensionless numbers (Re, f, relative roughness, friction slope)
# anchor calibration diagnostics
# method-comparison metrics
# discharge scenario analysis
# parameter sensitivity analysis
# uncertainty envelopes from Monte Carlo perturbations
# QA/QC checks for geometry formulas

# The flood pulse is implemented as repeated steady snapshots for scenario analysis.


import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal, List

import numpy as np
import pandas as pd


G = 9.81 # gravitational acceleration (m/s^2)




# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------
@dataclass
class FluidProps:
    rho: float = 1000.0
    nu: float = 1.0e-6  # m^2/s


@dataclass
class FlowCoeffs:
    ff:  float = 0.015    # Darcy friction factor from Blumberg-Curl (1974) / Springer (2004)
    n:   float = 0.045    # Manning roughness coefficient
    C:   float = 95.0     # Hazen-Williams roughness coefficient
    K_e: float = 0.5      # expansion loss coefficient (Jeannin 2001; White 2011)
    K_c: float = 0.3      # contraction loss coefficient (Jeannin 2001; White 2011)




@dataclass
class SensitivityConfig:
    discharge_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)
    manning_n_values: Tuple[float, ...] = (0.025, 0.035, 0.050, 0.060)
    hazen_C_values: Tuple[float, ...] = (90.0, 110.0, 130.0)
    ff_values:  Tuple[float, ...] = (0.010, 0.013, 0.020, 0.030)  # Darcy friction factor sensitivity
    Ke_values:  Tuple[float, ...] = (0.0, 0.3, 0.5, 1.0)
    Kc_values:  Tuple[float, ...] = (0.0, 0.2, 0.3, 0.5)
    monte_carlo_runs: int = 1000
    random_seed: int = 42


@dataclass
class InputPaths:
    survey_path: str = 'survey.txt'
    flow_table_path: str = 'Flow data ell.csv'
    scallop_path: str = 'Survey_and_scallop_data.csv'
    output_dir: str = 'outputs'


@dataclass
class Geometry:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    A: np.ndarray
    Rh: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.y = np.asarray(self.y, float)
        self.z = np.asarray(self.z, float)
        self.A = np.asarray(self.A, float)
        self.Rh = np.asarray(self.Rh, float)

        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dz = np.diff(self.z)
        ds = np.sqrt(dx * dx + dy * dy + dz * dz)
        self.s = np.zeros(len(self.x), float)
        self.s[1:] = np.cumsum(ds)
        self.De = 4.0 * np.maximum(self.Rh, 1e-12)



# --------------------------------------------------------------------
# Survey parsing + centerline
# --------------------------------------------------------------------
FT_TO_M = 0.3048
# This function reads the raw survey file line by line.
# It pulls out only the usable survey shots and ignore headers/noise.
# Each valid line should begin with an MFS (for this specific survey) station name and contain:
# from-station, to-station, shot length, azimuth, inclination, left, right, up, down.
# The survey file is stored in feet, so every distance value is converted to meters here.
# Output is a list of cleaned survey shots that the rest of the model can use.

def parse_survey_file(filename: str):
    # Read the survey text file and keep only the station-to-station
    # survey lines that begin with "MFS". Each accepted line becomes one shot
    # with length, bearing, inclination, and LRUD values.
    shots = []
    # Open the file safely and ignore odd characters so the parser does not crash.
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            # Collapse repeated spaces/tabs to one space so splitting is reliable.
            line = re.sub(r"\s+", " ", raw).strip()
            # Skip blank lines and anything that is not a survey shot.
            if not line or not line.startswith("MFS"):
                continue
            # Break the cleaned line into fields.
            parts = line.split(" ")
            # Require at least 9 fields: from, to, length, azimuth, inclination, L, R, U, D.
            if len(parts) < 9:
                continue
            try:
                frm, to = parts[0], parts[1]

                # survey.txt values are in feet so convert to meters
                length = float(parts[2]) * FT_TO_M
                bearing = float(parts[3])
                inc = float(parts[4])
                left = float(parts[5]) * FT_TO_M
                up = float(parts[6]) * FT_TO_M
                down = float(parts[7]) * FT_TO_M
                right = float(parts[8]) * FT_TO_M
                # Store one full survey shot as a tuple.
                shots.append((frm, to, length, bearing, inc, left, up, down, right))
            except Exception:
                continue
    print(f"Parsed {len(shots)} valid shots from {filename}")
    return shots

# This function turns the survey shots into a 3D centerline.
# For each shot:
#   - use azimuth + inclination to break the length into x, y, z parts
#   - add that step onto the previous station location
# This produces one XYZ point per station so the cave passage has a mappable path.
def survey_to_xyz(shots):
    # Convert each survey shot into Cartesian x, y, z coordinates by stepping
    # forward from the previous station using the shot length, azimuth, and inclination.
    pts = []
    x = y = z = 0.0
    #Start the centerline at an origin (0,0,0)
    pts.append((x, y, z))
    for frm, to, L, az, inc, left, up, down, right in shots:
        # Convert survey angles from degrees to radians for trig functions.
        azr = np.deg2rad(az)
        incr = np.deg2rad(inc)
        # Split the measured shot length into horizontal and vertical components.
        horiz = L * np.cos(incr)
        dz = L * np.sin(incr)
        # Convert the horizontal component into x/y map offsets using azimuth
        dx = horiz * np.sin(azr)
        dy = horiz * np.cos(azr)
        # Step from the previous station to the next one.
        x += dx
        y += dy
        z += dz
        #Starts the centerline at an origin (0,0,0).
        pts.append((x, y, z))
    return np.asarray(pts, float)


# --------------------------------------------------------------------
# LRUD geometry
# --------------------------------------------------------------------
#This block converts LRUD measurements into hydraulic geometry.
# LRUD's are treated as an ellipse for elliptical:
#   width  = left + right
#   height = up + down
#   area   = pi*a*b
def lrud_geometry_from_shots(shots) -> Tuple[np.ndarray, np.ndarray]:
    # Build arrays of left, right, up, and down measurements for every survey node
    Ls, Rs, Us, Ds = [], [], [], []
    if not shots:
        one = np.array([1.0], dtype=float)
        return one, one

    frm0, to0, L0_shot, az0, inc0, L0, U0, D0, R0 = shots[0]
    Ls.append(L0); Rs.append(R0); Us.append(U0); Ds.append(D0)

    for frm, to, L, az, inc, Lft, Up, Dn, Rgt in shots:
        Ls.append(Lft); Rs.append(Rgt); Us.append(Up); Ds.append(Dn)

    Ls = np.asarray(Ls, float); Rs = np.asarray(Rs, float)
    Us = np.asarray(Us, float); Ds = np.asarray(Ds, float)
    # Total passage width = left + right, and total height = up + down.
    # Tiny minimum values prevent divide-by-zero problems later.
    width  = np.maximum(Ls + Rs, 1e-12)
    height = np.maximum(Us + Ds, 1e-12)
    # a and b are the ellipse semi-axes.
    a = width  / 2.0
    b = height / 2.0

    # Compute cross-sectional area from the standard ellipse formula.
    A_lr = np.pi * a * b

    # Ramanujan perimeter approximation
    Aax = np.maximum(a, b)
    Bax = np.minimum(a, b)
    h = ((Aax - Bax) ** 2) / np.maximum((Aax + Bax) ** 2, 1e-12)
    P_ram = np.pi * (Aax + Bax) * (
            1 + h/4 + h**2/64 + h**3/256 + 25*h**4/16384 + 49*h**5/65536
    )

    # Hydraulic radius = A / P
    Rh_lr = A_lr / np.maximum(P_ram, 1e-12)

    return A_lr, Rh_lr

#--------------------------------------------------------------------
def load_flow_table(path_csv: str) -> pd.DataFrame:
    # Read the table that contains scallop-derived hydraulic values.
    return pd.read_csv(path_csv)

# --------------------------------------------------------------------
# Hydraulic equations
# --------------------------------------------------------------------
# Reynolds number block.
# This measures whether flow is laminar or turbulent.
# roughness / friction calculations.
def reynolds(V: np.ndarray, De: np.ndarray, nu: float) -> np.ndarray:
    return np.asarray(V, float) * np.asarray(De, float) / max(float(nu), 1e-12)

# Darcy-Weisbach head-loss block.
# Inputs are segment length, velocity, hydraulic diameter, and friction factor.
# Output is the frictional head loss over each passage segment.
# This is the main energy-loss equation for the Darcy version of the model.
def hf_darcy(ds: np.ndarray, V: np.ndarray, De: np.ndarray, f: np.ndarray) -> np.ndarray:
    ds = np.asarray(ds, float)
    V = np.asarray(V, float)
    De = np.maximum(np.asarray(De, float), 1e-12)
    f = np.asarray(f, float)
    return f * (ds / De) * (V * V) / (2.0 * G)

# Manning head-loss block.
# Computes the friction slope Sf from n, velocity, and hydraulic radius.
# Then multiply by segment length to get head loss over each segment.
# This gives the Manning-based energy-loss estimate for the cave passage.
def hf_manning(ds: np.ndarray, V: np.ndarray, Rh: np.ndarray, n: float) -> np.ndarray:
    ds = np.asarray(ds, float)
    V = np.asarray(V, float)
    Rh = np.maximum(np.asarray(Rh, float), 1e-12)
    Sf = (n * V / (Rh ** (2.0 / 3.0))) ** 2
    return Sf * ds

# Hazen-Williams head-loss block.
# Uses discharge and diameter instead of friction factor.
def hf_hazen(ds: np.ndarray, Q: np.ndarray, De: np.ndarray, C: float) -> np.ndarray:
    ds = np.asarray(ds, float)
    Q = np.asarray(Q, float)
    De = np.maximum(np.asarray(De, float), 1e-12)
    Sf = 10.67 * (Q ** 1.852) / ((C ** 1.852) * (De ** 4.87))
    return Sf * ds

# Swamee-Jain friction factor block.
# Estimates Darcy friction factor from Reynolds number and roughness.
# If Re is low, the code uses the laminar solution f = 64/Re.
# If Re is turbulent, it uses the Swamee-Jain explicit approximation.
# provides a starting guess for later friction calculations.
def swamee_jain_array(Re: np.ndarray, De: np.ndarray, k) -> np.ndarray:
    Re = np.asarray(Re, dtype=float)
    De = np.asarray(De, dtype=float)

    k = np.asarray(k, dtype=float)
    if k.ndim == 0:
        k = np.full_like(Re, float(k))

    f = np.zeros_like(Re, dtype=float)
    Re_safe = np.maximum(Re, 1.0)
    De_safe = np.maximum(De, 1e-12)
    # Use the laminar solution when Re < 2000
    lam = Re_safe < 2000.0
    f[lam] = 64.0 / Re_safe[lam]
    # Otherwise use the Swamee-Jain turbulent approximation.
    tur = ~lam
    x = (np.maximum(k[tur], 0.0) / (3.7 * De_safe[tur])) + (5.74 / (Re_safe[tur] ** 0.9))
    f[tur] = 0.25 / (np.log10(x) ** 2)
    return f


# --------------------------------------------------------------------
# Friction factor from scallops — Blumberg and Curl (1974)
# --------------------------------------------------------------------
# Scallop-based friction calibration block.
# Feed it derived scallop velocities and hydraulic diameters.
# For each scallop site, it solves the Blumberg & Curl equation
# to estimate the Darcy friction factor that is consistent with the scallops.
# The code starts with a reasonable guess for f, then iterates until the
# solution stops changing.
# Output is one friction factor per scallop site + one reach-average value
# that gets used across the full passage in the Darcy model.
def blumberg_curl_ff(
        V_anchor:  np.ndarray,
        De_anchor: np.ndarray,
        fluid:     FluidProps,
        n_iter:    int = 100,
) -> Tuple[float, pd.DataFrame]:

    n_sites = len(V_anchor)
    Re_arr  = np.zeros(n_sites, float)
    ff_arr  = np.zeros(n_sites, float)

    for j in range(n_sites):
        Re = float(max(
            V_anchor[j] * De_anchor[j] / max(fluid.nu, 1e-12), 1.0))
        Re_arr[j] = Re

        # Initial guess: Swamee-Jain smooth-wall estimate
        ff = float(swamee_jain_array(
            np.array([Re]), np.array([De_anchor[j]]),
            np.array([1e-9]))[0])

        # Iteratively solve ff^(-1/2) = 1.77*ln(Re*ff^(1/2)) - 11.44
        for _ in range(n_iter):
            sqrt_ff = np.sqrt(max(ff, 1e-12))
            rhs     = 1.77 * np.log(Re * sqrt_ff) - 11.44
            if abs(rhs) < 1e-12:
                break
            ff_new = max(1.0 / (rhs ** 2), 1e-6)
            if abs(ff_new - ff) < 1e-10:
                ff = ff_new
                break
            ff = ff_new

        ff_arr[j] = max(ff, 1e-6)

    ff_reach = float(np.mean(ff_arr))

    diag_df = pd.DataFrame({
        'anchor_site':      np.arange(1, n_sites + 1),
        'V_anchor_mps':     V_anchor,
        'De_anchor_m':      De_anchor,
        'Re_anchor':        Re_arr,
        'ff_blumberg_curl': ff_arr,
        'ff_reach_mean':    ff_reach,
    })
#Quick check to make usre it is solving for reasonable numbers
    print(f"  Blumberg-Curl ff per site: {np.round(ff_arr, 4)}")
    print(f"  Reach-averaged ff = {ff_reach:.4f}  "
          f"(Blumberg & Curl 1974)")

    return ff_reach, diag_df


# --------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------
# This is the core hydraulic model for one steady-flow run.
# Feed the cave geometry, one discharge Q, the roughness coefficients,
# and the method to use (darcy, manning, or hazen).
# The model treats the flood as one steady "snapshot" rather than
# a fully unsteady brief event.
# Steps inside this block:
# convert discharge into node velocities using V = Q/A
# average node values onto passage segments
# compute head loss by the chosen hydraulic equation
# add minor losses from expansions/contractions
# accumulate losses downstream to build HGL and EGL
# Output is one complete hydraulic profile along the passage.
def simulate_snapshot(
        geom: Geometry,
        Q: float,
        coeffs: FlowCoeffs,
        fluid: FluidProps,
        method: Literal['darcy', 'manning', 'hazen'] = 'darcy',
):
    # Convert discharge to velocity at each node using Q = V*A.
    V_nodes = Q / np.maximum(geom.A, 1e-12)
    V = 0.5 * (V_nodes[:-1] + V_nodes[1:])
    De = 0.5 * (geom.De[:-1] + geom.De[1:])
    Rh = 0.5 * (geom.Rh[:-1] + geom.Rh[1:])
    ds = np.sqrt(np.diff(geom.x) ** 2 + np.diff(geom.y) ** 2 + np.diff(geom.z) ** 2)
    # Compute Reynolds number so the flow regime can be determined.
    Re = reynolds(V, De, fluid.nu)

    #Chooses which hydraulic equation will provide the friction losses.
    if method == 'darcy':
        # Single reach-averaged friction factor from Blumberg & Curl (1974)
        # following Springer (2004): applied uniformly along the passage
        f = np.full_like(V, coeffs.ff)
        # Friction head loss for each segment.
        hf = hf_darcy(ds, V, De, f)
        Sf = hf / np.maximum(ds, 1e-12)
        rel_roughness = np.full_like(V, np.nan)   # not applicable with uniform ff

    elif method == 'manning':
        hf = hf_manning(ds, V, Rh, coeffs.n)
        Sf = hf / np.maximum(ds, 1e-12)
        f = np.full_like(V, np.nan)
        rel_roughness = np.full_like(V, np.nan)

    else:
        hf = hf_hazen(ds, np.full_like(ds, Q), De, coeffs.C)
        Sf = hf / np.maximum(ds, 1e-12)
        f = np.full_like(V, np.nan)
        rel_roughness = np.full_like(V, np.nan)
    # ------------------------------------------------------------------
    # Minor losses (expansion and contraction)
    # ------------------------------------------------------------------
    # At each node, compare V_nodes[i] (upstream) to V_nodes[i+1]
    # (downstream):
    #expansion  (A increasing, V decreasing): h_e = K_e*(V1-V2)^2 / 2g
    #contraction (A decreasing, V increasing): h_c = K_c * V2^2 / 2g
    dV      = np.diff(V_nodes)                      # V[i+1] - V[i], length n_segs
    V_up    = V_nodes[:-1]
    V_down  = V_nodes[1:]

    expanding   = dV < 0 # velocity decreasing = area expanding
    contracting = dV > 0 # velocity increasing = area contracting

    hf_minor = np.zeros_like(hf)
    hf_minor[expanding]   = coeffs.K_e * (V_up[expanding]  - V_down[expanding])**2  / (2 * G)
    hf_minor[contracting] = coeffs.K_c *  V_down[contracting]**2                    / (2 * G)
    # Total loss is the sum of friction loss and minor loss.
    hf_total = hf + hf_minor
    # Cumulative head loss is subtracted from the upstream head to get the HGL.
    hf_cum = np.concatenate([[0.0], np.cumsum(hf_total)])   # use hf_total, not hf
    HGL = geom.z[0] - hf_cum
    # Energy grade line adds velocity head to the hydraulic grade line.
    EGL = HGL + V_nodes ** 2 / (2 * G)

    # Expand segment diagnostics to node-length arrays for export reasons
    Re_nodes        = np.concatenate([[Re[0]],            Re])           if len(Re)            else np.array([np.nan])
    f_nodes         = np.concatenate([[f[0]],             f])            if len(f)             else np.array([np.nan])
    Sf_nodes        = np.concatenate([[Sf[0]],            Sf])           if len(Sf)            else np.array([np.nan])
    rel_rough_nodes = np.concatenate([[rel_roughness[0]], rel_roughness]) if len(rel_roughness) else np.array([np.nan])
    hf_minor_nodes  = np.concatenate([[hf_minor[0]],      hf_minor])     if len(hf_minor)      else np.array([np.nan])

    return {
        's':                  geom.s,
        'z':                  geom.z,
        'HGL':                HGL,
        'EGL':                EGL,
        'V':                  V_nodes,
        'Re':                 Re_nodes[:len(geom.s)],
        'f':                  f_nodes[:len(geom.s)],
        'Sf':                 Sf_nodes[:len(geom.s)],
        'relative_roughness': rel_rough_nodes[:len(geom.s)],
        'hf_minor':           hf_minor_nodes[:len(geom.s)],
        'Q':                  np.full(len(geom.s), Q),
    }

# Simple hydrograph generator.
# Makes a triangular flood pulse.
# Discharge rises from base flow to peak flow, then falls back down.
# Provides a sequence of steady discharges for repeated snapshot runs.
def make_hydrograph(Q_base: float, Q_peak: float, n_steps: int):
    # Create a simple triangular flood hydrograph that rises to a peak and then falls.
    t = np.linspace(0.0, 1.0, n_steps)
    mid = n_steps // 2
    Q_t = np.ones(n_steps, float) * Q_base
    for i in range(n_steps):
        if i <= mid:
            Q_t[i] = Q_base + (Q_peak - Q_base) * (i / max(mid, 1))
        else:
            Q_t[i] = Q_peak - (Q_peak - Q_base) * ((i - mid) / max(n_steps - mid - 1, 1))
    return t, Q_t

# Flood-pulse block.
# Feed it the geometry and a range of discharges through time.
# For every time step and every hydraulic method, it runs one steady snapshot.
# Output is time-series arrays that can later be animated or compared.
def compute_flood_pulse_snapshots(
        geom: Geometry,
        fluid: FluidProps,
        coeffs: FlowCoeffs,
        methods=('darcy', 'manning', 'hazen'),
        Q_base=0.5,
        Q_peak=3.0,
        n_steps=40,
):
    t, Q_t = make_hydrograph(Q_base, Q_peak, n_steps)
    n_nodes = len(geom.x)
    snapshots: Dict[str, Dict[str, np.ndarray]] = {}
    # Pre-allocate arrays to store HGL, EGL, and velocity for every time step.
    for m in methods:
        snapshots[m] = {'HGL': np.zeros((n_steps, n_nodes)), 'EGL': np.zeros((n_steps, n_nodes)), 'V': np.zeros((n_steps, n_nodes))}
    # Run a steady-state snapshot at each hydrograph step.
    for it in range(n_steps):
        Q_now = float(Q_t[it])
        # Pre-allocate arrays to store HGL, EGL, and velocity for every time step.
        for m in methods:
            snap = simulate_snapshot(geom, Q_now, coeffs, fluid, method=m)
            for key in ['HGL', 'EGL', 'V']:
                arr = np.asarray(snap[key], dtype=float)
                snapshots[m][key][it, :] = arr[:n_nodes]

    return t, Q_t, snapshots


# --------------------------------------------------------------------
# Diagnostics and thesis exports
# --------------------------------------------------------------------
def qa_geometry_check(A_lr: np.ndarray, Rh_lr: np.ndarray, P_lr: np.ndarray) -> pd.DataFrame:
    Rh_from_AP = A_lr / np.maximum(P_lr, 1e-12)
    abs_diff = np.abs(Rh_lr - Rh_from_AP)
    rel_diff_pct = 100.0 * abs_diff / np.maximum(np.abs(Rh_from_AP), 1e-12)
    return pd.DataFrame(
        {
            'A_lr_m2': A_lr,
            'Rh_curl_m': Rh_lr,
            'Rh_A_over_P_m': Rh_from_AP,
            'abs_diff_m': abs_diff,
            'rel_diff_pct': rel_diff_pct,
        }
    )


def summarize_geometry(geom: Geometry, A_lr: np.ndarray, Rh_lr: np.ndarray, sA: np.ndarray, sR: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            's_m': geom.s,
            'x_m': geom.x,
            'y_m': geom.y,
            'z_m': geom.z,
            'A_lr_m2': A_lr,
            'Rh_lr_m': Rh_lr,
            'A_scaled_m2': geom.A,
            'Rh_scaled_m': geom.Rh,
            'De_scaled_m': geom.De,
            'scale_A': sA,
            'scale_Rh': sR,
        }
    )


def build_anchor_geometry_diagnostics(
        geom: Geometry,
        A_lr: np.ndarray,
        Rh_lr: np.ndarray,
        anchor_idx: np.ndarray,
        anchor_names: List[str],
        A_anchor: np.ndarray,
        Rh_anchor: np.ndarray,
        sA: np.ndarray,
        sR: np.ndarray,
) -> pd.DataFrame:
    idx = np.asarray(anchor_idx, int)
    return pd.DataFrame(
        {
            'anchor_name': anchor_names,
            'node_idx': idx,
            's_m': geom.s[idx],
            'A_anchor_measured_m2': A_anchor,
            'A_lr_raw_m2': A_lr[idx],
            'A_scaled_m2': geom.A[idx],
            'A_scale_factor': sA[idx],
            'Rh_anchor_measured_m': Rh_anchor,
            'Rh_lr_raw_m': Rh_lr[idx],
            'Rh_scaled_m': geom.Rh[idx],
            'Rh_scale_factor': sR[idx],
        }
    )


def summary_metrics_for_snapshot(method: str, snap: Dict[str, np.ndarray], geom: Geometry) -> Dict[str, float]:
    HGL = np.asarray(snap['HGL'], float)
    EGL = np.asarray(snap['EGL'], float)
    V = np.asarray(snap['V'], float)
    Re = np.asarray(snap['Re'], float)
    Sf = np.asarray(snap['Sf'], float)

    total_headloss = float(HGL[0] - HGL[-1])
    pressure_head = HGL - geom.z
    energy_head = EGL - geom.z

    return {
        'method': method,
        'Q_m3s': float(np.nanmedian(snap['Q'])),
        'total_headloss_m': total_headloss,
        'mean_velocity_mps': float(np.nanmean(V)),
        'max_velocity_mps': float(np.nanmax(V)),
        'mean_Re': float(np.nanmean(Re)),
        'max_Re': float(np.nanmax(Re)),
        'mean_pressure_head_m': float(np.nanmean(pressure_head)),
        'max_pressure_head_m': float(np.nanmax(pressure_head)),
        'mean_energy_head_m': float(np.nanmean(energy_head)),
        'max_energy_head_m': float(np.nanmax(energy_head)),
        'mean_friction_slope': float(np.nanmean(Sf)),
        'max_friction_slope': float(np.nanmax(Sf)),
    }


def compare_methods(results: Dict[str, Dict[str, np.ndarray]], base: str = 'darcy') -> pd.DataFrame:
    base_HGL = np.asarray(results[base]['HGL'], float)
    rows = []
    for method, snap in results.items():
        H = np.asarray(snap['HGL'], float)
        dH = H - base_HGL
        rows.append(
            {
                'method': method,
                'base_method': base,
                'rms_delta_HGL_m': float(np.sqrt(np.mean(dH ** 2))),
                'max_abs_delta_HGL_m': float(np.max(np.abs(dH))),
                'end_delta_HGL_m': float(dH[-1]),
                'mean_delta_HGL_m': float(np.mean(dH)),
            }
        )
    return pd.DataFrame(rows)


def build_node_export(method: str, snap: Dict[str, np.ndarray], geom: Geometry) -> pd.DataFrame:
    # Build one table that stores the full node-by-node hydraulic result for a method.
    return pd.DataFrame({
        'method':             method,
        's_m':                geom.s,
        'x_m':                geom.x,
        'y_m':                geom.y,
        'z_m':                geom.z,
        'A_m2':               geom.A,
        'Rh_m':               geom.Rh,
        'De_m':               geom.De,
        'Q_m3s':              np.asarray(snap['Q'],    float),
        'HGL_m':              np.asarray(snap['HGL'],  float),
        'EGL_m':              np.asarray(snap['EGL'],  float),
        'V_mps':              np.asarray(snap['V'],    float),
        'Re':                 np.asarray(snap['Re'],   float),
        'f_darcy':            np.asarray(snap['f'],    float),
        'Sf':                 np.asarray(snap['Sf'],   float),
        'relative_roughness': np.asarray(snap['relative_roughness'], float),
        'hf_minor_m':         np.asarray(snap['hf_minor'], float),
    })


def build_anchor_hydraulic_summary(
        method: str,
        snap: Dict[str, np.ndarray],
        anchor_idx: np.ndarray,
        anchor_names: List[str],
        V_anchor: np.ndarray,
        roughness_diag: pd.DataFrame,
) -> pd.DataFrame:
    idx = np.asarray(anchor_idx, int)
    out = pd.DataFrame({
        'method':             method,
        'anchor_name':        anchor_names,
        'node_idx':           idx,
        's_m':                np.asarray(snap['s'],   float)[idx],
        'HGL_m':              np.asarray(snap['HGL'], float)[idx],
        'EGL_m':              np.asarray(snap['EGL'], float)[idx],
        'V_model_mps':        np.asarray(snap['V'],   float)[idx],
        'V_anchor_input_mps': V_anchor,
        'Re_model':           np.asarray(snap['Re'],  float)[idx],
        'Sf_model':           np.asarray(snap['Sf'],  float)[idx],
        'f_model':            np.asarray(snap['f'],   float)[idx],
    })
    if method == 'darcy' and 'ff_blumberg_curl' in roughness_diag.columns:
        out['ff_blumberg_curl'] = roughness_diag['ff_blumberg_curl'].values
    return out

# --------------------------------------------------------------------
# Scenario sweep for different discharges.
# --------------------------------------------------------------------
# This tests how sensitive each method is to changing Q.
# The same geometry is kept fixed while discharge is scaled up or down.
# Output is a table of summary metrics for each scenario.
def run_discharge_scenarios(
        geom: Geometry,
        fluid: FluidProps,
        coeffs: FlowCoeffs,
        Q0: float,
        discharge_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
) -> pd.DataFrame:
    rows = []
    for qmult in discharge_multipliers:
        Q = Q0 * qmult
        for method in ['darcy', 'manning', 'hazen']:
            snap = simulate_snapshot(geom, Q, coeffs, fluid, method=method)
            row = summary_metrics_for_snapshot(method, snap, geom)
            row['scenario_type'] = 'discharge'
            row['scenario_value'] = qmult
            rows.append(row)
    return pd.DataFrame(rows)

# Parameter sensitivity block.
# This asks how much do the model results change if roughness assumptions change?
# It repeats the model while varying Manning n, Hazen-Williams C,
# Darcy friction factor, and minor-loss coefficients.
# Output is tables that show which parameters most affect head loss.
def run_parameter_sensitivity(
        geom:   Geometry,
        fluid:  FluidProps,
        coeffs: FlowCoeffs,
        Q0:     float,
        cfg:    SensitivityConfig,
) -> pd.DataFrame:
    rows = []

    # Manning n sensitivity
    for nval in cfg.manning_n_values:
        c    = FlowCoeffs(ff=coeffs.ff, n=float(nval), C=coeffs.C,
                          K_e=coeffs.K_e, K_c=coeffs.K_c)
        snap = simulate_snapshot(geom, Q0, c, fluid, method='manning')
        row  = summary_metrics_for_snapshot('manning', snap, geom)
        row['scenario_type']  = 'manning_n'
        row['scenario_value'] = float(nval)
        rows.append(row)

    # Hazen-Williams C sensitivity
    for cval in cfg.hazen_C_values:
        c    = FlowCoeffs(ff=coeffs.ff, n=coeffs.n, C=float(cval),
                          K_e=coeffs.K_e, K_c=coeffs.K_c)
        snap = simulate_snapshot(geom, Q0, c, fluid, method='hazen')
        row  = summary_metrics_for_snapshot('hazen', snap, geom)
        row['scenario_type']  = 'hazen_C'
        row['scenario_value'] = float(cval)
        rows.append(row)

    # Darcy friction factor sensitivity (Blumberg-Curl reach ff +/- range)
    for ffval in cfg.ff_values:
        c    = FlowCoeffs(ff=float(ffval), n=coeffs.n, C=coeffs.C,
                          K_e=coeffs.K_e, K_c=coeffs.K_c)
        snap = simulate_snapshot(geom, Q0, c, fluid, method='darcy')
        row  = summary_metrics_for_snapshot('darcy', snap, geom)
        row['scenario_type']  = 'darcy_ff'
        row['scenario_value'] = float(ffval)
        rows.append(row)

    # Expansion loss coefficient sensitivity (all three methods)
    for Ke_val in cfg.Ke_values:
        for method in ['darcy', 'manning', 'hazen']:
            c    = FlowCoeffs(ff=coeffs.ff, n=coeffs.n, C=coeffs.C,
                              K_e=float(Ke_val), K_c=coeffs.K_c)
            snap = simulate_snapshot(geom, Q0, c, fluid, method=method)
            row  = summary_metrics_for_snapshot(method, snap, geom)
            row['scenario_type']  = 'K_e'
            row['scenario_value'] = float(Ke_val)
            rows.append(row)

    # Contraction loss coefficient sensitivity (all three methods)
    for Kc_val in cfg.Kc_values:
        for method in ['darcy', 'manning', 'hazen']:
            c    = FlowCoeffs(ff=coeffs.ff, n=coeffs.n, C=coeffs.C,
                              K_e=coeffs.K_e, K_c=float(Kc_val))
            snap = simulate_snapshot(geom, Q0, c, fluid, method=method)
            row  = summary_metrics_for_snapshot(method, snap, geom)
            row['scenario_type']  = 'K_c'
            row['scenario_value'] = float(Kc_val)
            rows.append(row)

    return pd.DataFrame(rows)

# --------------------------------------------------------------------
# Monte Carlo Uncertainity
# --------------------------------------------------------------------
# Monte Carlo uncertainty block.
# Quantifies how much the output could shift if geometry and discharge
# are slightly wrong.
# For each run, the code perturbs area, hydraulic radius, and discharge by
# small random amounts, reruns the model, and stores the result.
# Output = p05 / p50 / p95 envelopes for HGL and velocity.
def monte_carlo_uncertainty_envelopes(
        geom_base: Geometry,
        fluid: FluidProps,
        coeffs: FlowCoeffs,
        Q0: float,
        n_runs: int = 1000,
        seed: int = 42,
        area_sigma_frac: float = 0.05,
        rh_sigma_frac: float = 0.05,
        velocity_sigma_frac: float = 0.05,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    rng     = np.random.default_rng(seed)
    methods = ['darcy', 'manning', 'hazen']
    storage = {m: {'HGL': [], 'V': [], 'headloss': []} for m in methods}

    for i in range(n_runs):
        if (i + 1) % 200 == 0:
            print(f"  MC run {i+1}/{n_runs}")
        A_pert  = geom_base.A  * np.clip(1.0 + rng.normal(0.0, area_sigma_frac,  size=len(geom_base.A)),  0.5, 1.5)
        Rh_pert = geom_base.Rh * np.clip(1.0 + rng.normal(0.0, rh_sigma_frac,    size=len(geom_base.Rh)), 0.5, 1.5)
        Q_pert  = float(Q0 * np.clip(1.0 + rng.normal(0.0, velocity_sigma_frac), 0.5, 1.5))

        geom = Geometry(geom_base.x, geom_base.y, geom_base.z, A_pert, Rh_pert)

        for method in methods:
            snap = simulate_snapshot(geom, Q_pert, coeffs, fluid, method=method)
            HGL  = np.asarray(snap['HGL'], float)
            storage[method]['HGL'].append(HGL)
            storage[method]['V'].append(np.asarray(snap['V'], float))
            storage[method]['headloss'].append(float(HGL[0] - HGL[-1]))

    envelope_dfs           = {}
    headloss_distributions = {}
    for method in methods:
        H = np.asarray(storage[method]['HGL'], float)
        V = np.asarray(storage[method]['V'],   float)
        envelope_dfs[method] = pd.DataFrame({
            's_m':       geom_base.s,
            'HGL_p05_m': np.nanpercentile(H, 5,  axis=0),
            'HGL_p50_m': np.nanpercentile(H, 50, axis=0),
            'HGL_p95_m': np.nanpercentile(H, 95, axis=0),
            'V_p05_mps': np.nanpercentile(V, 5,  axis=0),
            'V_p50_mps': np.nanpercentile(V, 50, axis=0),
            'V_p95_mps': np.nanpercentile(V, 95, axis=0),
        })
        headloss_distributions[method] = np.array(storage[method]['headloss'])

    return envelope_dfs, headloss_distributions



# --------------------------------------------------------------------
# Input/Output helpers
# --------------------------------------------------------------------
def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_velocity_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if 'velocity' in str(c).lower():
            return c
    raise KeyError(f"No velocity column found. Columns={list(df.columns)}")


def save_json(path: Path, payload: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------
# Validation against measured velocities
# --------------------------------------------------------------------

@dataclass
class ValidationData:
    # Stores observed velocity data for model validation
    # Each row represents one field measurement location:
    #   s_obs  = distance along the cave passage (m)
    #   V_obs  = measured average velocity at that location (m/s)
    #   labels = name or ID of the measurement site (e.g., "Scallop Site 1")
    #
    s_obs:  np.ndarray
    V_obs:  np.ndarray
    labels: List[str]

    def __post_init__(self):
        self.s_obs = np.asarray(self.s_obs, float)
        self.V_obs = np.asarray(self.V_obs, float)
        if len(self.s_obs) != len(self.V_obs):
            raise ValueError("s_obs and V_obs must have the same length.")
        if len(self.labels) != len(self.s_obs):
            raise ValueError("labels must have the same length as s_obs.")

    @classmethod
    def from_csv(cls, path: str) -> 'ValidationData':

        #Load validation points from a CSV file.
        #Required columns: s_m, V_obs_mps
        

        df = pd.read_csv(path)
        required = {'s_m', 'V_obs_mps'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Validation CSV is missing columns: {missing}. "
                f"Found: {list(df.columns)}"
            )
        labels = (
            df['label'].tolist()
            if 'label' in df.columns
            else [f'Site {i+1}' for i in range(len(df))]
        )
        return cls(
            s_obs=df['s_m'].to_numpy(float),
            V_obs=df['V_obs_mps'].to_numpy(float),
            labels=labels,
        )


def interpolate_modeled_velocity(
        s_nodes: np.ndarray,
        V_nodes: np.ndarray,
        s_obs: np.ndarray,
) -> np.ndarray:

    #Linearly interpolate the modeled velocity profile to the observed
    #measurement locations.

    return np.interp(s_obs, s_nodes, V_nodes)


def compute_validation_metrics(
        V_obs: np.ndarray,
        V_mod: np.ndarray,
) -> dict:

    #Compute standard goodness-of-fit metrics between observed and
    #modeled velocities.

    #Metrics returned:
    #RMSE  : Root Mean Square Error (m/s)
    #MAE   : Mean Absolute Error (m/s)
    #PBIAS : Percent Bias (%) — positive = model overestimates
    #NSE   : Nash–Sutcliffe Efficiency (1 = perfect, <0 = worse than mean)
    #R2    : Pearson coefficient of determination

    V_obs = np.asarray(V_obs, float)
    V_mod = np.asarray(V_mod, float)
    n     = len(V_obs)

    residuals   = V_mod - V_obs
    ss_res      = float(np.sum(residuals ** 2))
    ss_tot      = float(np.sum((V_obs - np.mean(V_obs)) ** 2))

    rmse  = float(np.sqrt(np.mean(residuals ** 2)))
    mae   = float(np.mean(np.abs(residuals)))
    pbias = float(100.0 * np.sum(residuals) / max(np.sum(V_obs), 1e-12))
    nse   = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Pearson R²
    if n > 1 and np.std(V_obs) > 1e-12 and np.std(V_mod) > 1e-12:
        r2 = float(np.corrcoef(V_obs, V_mod)[0, 1] ** 2)
    else:
        r2 = float('nan')

    return {
        'RMSE_mps':  rmse,
        'MAE_mps':   mae,
        'PBIAS_pct': pbias,
        'NSE':       nse,
        'R2':        r2,
    }


# --------------------------------------------------------------------
# Validation — discharge closure test
# --------------------------------------------------------------------
# Validation block for instacnes where measured hydraulic head is not available.
# Because V = Q/A is shared across methods, direct velocity comparison does not
# separate Darcy, Manning, and Hazen very well.
# Instead this block back-calculates discharge from each method's own
# friction slope, then compares that discharge to the discharge implied by
# the scallop-derived observed velocity.
def run_discharge_closure_validation(
        results:   dict,
        val_data:  ValidationData,
        geom:      'Geometry',
        coeffs:    'FlowCoeffs',
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    point_rows   = []
    metrics_rows = []

    # Observed discharge at each validation site: Q_obs = V_obs * A_site
    A_sites = np.interp(val_data.s_obs, geom.s, geom.A)
    Q_obs   = val_data.V_obs * A_sites

    for method, snap in results.items():
        s_nodes  = np.asarray(snap['s'],  float)
        Sf_nodes = np.asarray(snap['Sf'], float)
        f_nodes  = np.asarray(snap['f'],  float)
        Rh_nodes = geom.Rh
        De_nodes = geom.De

        # Back-calculate velocity from each method's own Sf
        if method == 'darcy':
            # hf = f*(Δs/De)*V²/(2g)  →  Sf = f*V²/(De*2g)
            # to V = sqrt(Sf * De * 2g / f)
            f_safe = np.where(
                np.isfinite(f_nodes) & (f_nodes > 1e-12),
                f_nodes, np.nan
            )
            V_back = np.sqrt(
                np.maximum(Sf_nodes * De_nodes * 2.0 * G, 0.0) / f_safe
            )

        elif method == 'manning':
            # Sf = (nV / Rh^(2/3))²  →  V = (1/n) * Rh^(2/3) * sqrt(Sf)
            V_back = (1.0 / coeffs.n) * (
                    np.maximum(Rh_nodes, 1e-12) ** (2.0 / 3.0)
            ) * np.sqrt(np.maximum(Sf_nodes, 0.0))

        else:  # hazen-williams
            # Sf = 10.67*Q^1.852 / (C^1.852 * De^4.87)
            # Q = (Sf * C^1.852 * De^4.87 / 10.67)^(1/1.852)
            # V = Q / A
            Q_back_nodes = (
                                   np.maximum(Sf_nodes, 0.0)
                                   * (coeffs.C ** 1.852)
                                   * (np.maximum(De_nodes, 1e-12) ** 4.87)
                                   / 10.67
                           ) ** (1.0 / 1.852)
            V_back = Q_back_nodes / np.maximum(geom.A, 1e-12)

        # Replace any non-finite back-calculated values with Q/A forward velocity
        V_forward = np.asarray(snap['V'], float)
        V_back = np.where(np.isfinite(V_back), V_back, V_forward)

        # Interpolate back-calculated velocity to observed sites
        V_back_sites = np.interp(val_data.s_obs, s_nodes, V_back)
        Q_back_sites = V_back_sites * A_sites

        #Per-site rows
        for i, label in enumerate(val_data.labels):
            point_rows.append({
                'method':              method,
                'site':                label,
                's_obs_m':             val_data.s_obs[i],
                'A_site_m2':           A_sites[i],
                'V_obs_mps':           val_data.V_obs[i],
                'Q_obs_m3s':           Q_obs[i],
                'V_back_mps':          V_back_sites[i],
                'Q_back_m3s':          Q_back_sites[i],
                'V_residual_mps':      V_back_sites[i] - val_data.V_obs[i],
                'Q_residual_m3s':      Q_back_sites[i] - Q_obs[i],
                'Q_pct_error':         100.0 * (Q_back_sites[i] - Q_obs[i])
                                       / max(Q_obs[i], 1e-12),
            })

        #Summary metrics on discharge
        metrics = compute_validation_metrics(Q_obs, Q_back_sites)
        metrics_rows.append({
            'method':           method,
            'validation_type':  'discharge_closure',
            'n_sites':          len(Q_obs),
            'Q_obs_median_m3s': float(np.median(Q_obs)),
            **metrics,
        })

    point_df   = pd.DataFrame(point_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    return point_df, metrics_df

def summarize_geometry_simple(geom: Geometry) -> pd.DataFrame:
    #Simple geometry summary using LRUD ellipses directly.
    return pd.DataFrame({
        's_m':   geom.s,
        'x_m':   geom.x,
        'y_m':   geom.y,
        'z_m':   geom.z,
        'A_m2':  geom.A,
        'Rh_m':  geom.Rh,
        'De_m':  geom.De,
    })
# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------
# Main function that ties the above ^ together.
# Order of opertaions:
#   1) read all input files
#   2) build the survey centerline
#   3) convert LRUD into hydraulic geometry
#   4) estimate baseline discharge from scallops
#   5) calibrate Darcy friction factor from scallop anchor points
#   6) run Darcy, Manning, and Hazen snapshots
#   7) validate results against observed/scallop-derived data
#   8) export node tables, summaries, and diagnostics
#   9) run scenarios, uncertainty, and flood-pulse analyses

def run_full_model(
        paths: InputPaths,
        station_map: Optional[Dict[str, str]] = None,
        coeffs: Optional[FlowCoeffs] = None,
        fluid: Optional[FluidProps] = None,
        sens: Optional[SensitivityConfig] = None,
        show_plots: bool = False,
        validation_path: Optional[str] = None,
):
    fluid  = fluid or FluidProps()
    sens   = sens  or SensitivityConfig()
    station_map = station_map or {'Model 1': 'MFS5', 'Model 2': 'MFS8', 'Model 3': 'MFS14', 'Model 4': 'MFS20'}

    outdir = ensure_output_dir(paths.output_dir)

    # 1) Inputs
    shots = parse_survey_file(paths.survey_path)
    if len(shots) < 5:
        raise SystemExit("Not enough shots parsed — check parser/filename.")

    centerline = survey_to_xyz(shots)
    x, y, z = centerline[:, 0], centerline[:, 1], centerline[:, 2]
    n_nodes = len(x)

    A_lr, Rh_lr = lrud_geometry_from_shots(shots)
    flow_df = load_flow_table(paths.flow_table_path)

    anchor_names = list(station_map.values())
    anchor_labels = list(station_map.keys())
    anchor_idx = np.array([max(0, min(n_nodes - 1, int(s.replace('MFS', '')) - 1)) for s in anchor_names], dtype=int)

    # Geometry comes directly from LRUD ellipses — no anchor scaling applied.
    # Area = pi*a*b, Rh = A/P following Springer (2004).
    
    A_nodes = A_lr[:n_nodes]
    Rh_nodes = Rh_lr[:n_nodes]
    geom = Geometry(x=x, y=y, z=z, A=A_nodes, Rh=Rh_nodes)

# 2) Baseline discharge from scallops
    df_sc = pd.read_csv(paths.scallop_path)
    vel_col = find_velocity_column(df_sc)
    V_excel = df_sc[vel_col].to_numpy(float)
    scallop_rows = np.where(np.isfinite(V_excel) & (V_excel > 0))[0]
    scallop_nodes = scallop_rows + 1
    scallop_nodes = scallop_nodes[scallop_nodes < n_nodes]
    if len(scallop_nodes) == 0:
        raise ValueError("No scallop sites found (Velocity column has no finite > 0 values).")

    Qi_sites = V_excel[scallop_rows[:len(scallop_nodes)]] * A_nodes[scallop_nodes]
    Qi_sites = Qi_sites[np.isfinite(Qi_sites) & (Qi_sites > 0)]
    Q0 = float(np.median(Qi_sites))
    # 3) Friction factor from scallops — Blumberg & Curl (1974) / Springer (2004)
    if 'Velocity' in flow_df.columns:
        V_anchor = flow_df['Velocity'].to_numpy(float)
    else:
        V_anchor = np.array([Q0 / max(A_nodes[i], 1e-12) for i in anchor_idx], dtype=float)

    De_anchor = geom.De[anchor_idx]
    ff_reach, roughness_diag = blumberg_curl_ff(V_anchor, De_anchor, fluid)
    roughness_diag.to_csv(outdir / 'darcy_roughness_calibration.csv', index=False)

    # Build FlowCoeffs with calibrated ff
    if coeffs is None:
        coeffs = FlowCoeffs(ff=ff_reach)
    else:
        coeffs = FlowCoeffs(ff=ff_reach, n=coeffs.n, C=coeffs.C,
                            K_e=coeffs.K_e, K_c=coeffs.K_c)

    # 4) Snapshot runs
    results = {}
    for method in ['darcy', 'manning', 'hazen']:
        results[method] = simulate_snapshot(geom, Q0, coeffs, fluid, method=method)


    #4b validate model using observed/scallop-derived velocties
    val_data = None
    if validation_path is not None and Path(validation_path).exists():
        val_data = ValidationData.from_csv(validation_path)
    elif validation_path is not None:
        print(f"[Validation] '{validation_path}' not found — falling back to scallop sites.")

    if val_data is None:
        val_labels = [f'Scallop Site {i+1}' for i in range(len(scallop_nodes))]
        val_data = ValidationData(
            s_obs  = geom.s[scallop_nodes],
            V_obs  = V_excel[scallop_rows[:len(scallop_nodes)]],
            labels = val_labels,
        )

    validation_point_df, validation_metrics_df = run_discharge_closure_validation(
        results, val_data, geom, coeffs
    )
    #Save validation outputs
    validation_point_df.to_csv(outdir / 'validation_point_comparison.csv',   index=False)
    validation_metrics_df.to_csv(outdir / 'validation_metrics_summary.csv',  index=False)

    print("\n── Discharge closure validation ────────────────────────")
    print("   Metrics compare back-calculated Q (from each method's")
    print("   own friction slope) against scallop-observed Q = V*A")
    print(validation_metrics_df[
              ['method', 'RMSE_mps', 'MAE_mps', 'PBIAS_pct', 'NSE', 'R2']
          ].rename(columns={
        'RMSE_mps': 'RMSE (m³/s)',
        'MAE_mps':  'MAE (m³/s)',
    }).to_string(index=False))
    print("────────────────────────────────────────────────────────\n")

    # 5) Export geometry + QA + anchor diagnostics + node tables
    qa_df = pd.DataFrame({
        'node': np.arange(len(A_nodes)),
        's_m': geom.s,
        'A_m2': A_nodes,
        'Rh_m': Rh_nodes,
        'De_m': geom.De,
    })
    geom_df = summarize_geometry_simple(geom)

    qa_df.to_csv(outdir / 'geometry_qaqc.csv', index=False)
    geom_df.to_csv(outdir / 'geometry_nodes.csv', index=False)

    method_summary_rows = []
    for method, snap in results.items():
        node_df = build_node_export(method, snap, geom)
        node_df.to_csv(outdir / f'{method}_nodes.csv', index=False)

        anchor_df = build_anchor_hydraulic_summary(
            method, snap, anchor_idx, anchor_names, V_anchor, roughness_diag
        )
        anchor_df.to_csv(outdir / f'{method}_anchor_summary.csv', index=False)
        method_summary_rows.append(summary_metrics_for_snapshot(method, snap, geom))

    method_summary_df = pd.DataFrame(method_summary_rows)
    method_compare_df = compare_methods(results, base='darcy')
    method_summary_df.to_csv(outdir / 'method_summary_metrics.csv', index=False)
    method_compare_df.to_csv(outdir / 'method_comparison_metrics.csv', index=False)

    # 6) Scenario analysis
    discharge_df = run_discharge_scenarios(geom, fluid, coeffs, Q0, sens.discharge_multipliers)
    sensitivity_df = run_parameter_sensitivity(geom, fluid, coeffs, Q0, sens)
    discharge_df.to_csv(outdir / 'scenario_discharge_sweep.csv', index=False)
    sensitivity_df.to_csv(outdir / 'scenario_parameter_sensitivity.csv', index=False)

    # 7) Uncertainty envelopes
    uncertainty, headloss_dists = monte_carlo_uncertainty_envelopes(
        geom, fluid, coeffs, Q0,
        n_runs=sens.monte_carlo_runs,
        seed=sens.random_seed,
    )
    for method, df in uncertainty.items():
        df.to_csv(outdir / f'{method}_uncertainty_envelope.csv', index=False)
    pd.DataFrame(headloss_dists).to_csv(outdir / 'mc_headloss_distributions.csv', index=False)


    # 8) Metadata export
    metadata = {
        'Q0_m3s': Q0,
        'anchor_names': anchor_names,
        'anchor_labels': anchor_labels,
        'anchor_idx': anchor_idx.tolist(),
        'scallop_rows': scallop_rows.tolist(),
        'scallop_nodes': scallop_nodes.tolist(),
        'ff_reach': ff_reach,
        'coeffs': asdict(coeffs),
        'fluid': asdict(fluid),
        'sensitivity': asdict(sens),
    }
    save_json(outdir / 'run_metadata.json', metadata)



    print(f"Finished. Outputs saved to: {outdir.resolve()}")
    print(f"Baseline discharge Q0   = {Q0:.6f} m^3/s")
    print(f"Reach-averaged ff (B&C) = {ff_reach:.4f}")
    print("Key exports:")
    print("  - geometry_nodes.csv")
    print("  - darcy_roughness_calibration.csv  (Blumberg-Curl per site)")
    print("  - mc_headloss_distributions.csv")
    print("  - method_summary_metrics.csv")
    print("  - method_comparison_metrics.csv")
    print("  - scenario_discharge_sweep.csv")
    print("  - scenario_parameter_sensitivity.csv")
    print("  - *_uncertainty_envelope.csv")
    print("  - *_nodes.csv and *_anchor_summary.csv")

    return {
        'results':          results,
        'uncertainty':      uncertainty,
        'headloss_dists':   headloss_dists,
        'discharge_df':     discharge_df,
        'sensitivity_df':   sensitivity_df,
        'method_summary_df': method_summary_df,
        'method_compare_df': method_compare_df,
        'roughness_diag':   roughness_diag,
        'metadata':         metadata,
        'Q0':               Q0,
        'ff_reach':         ff_reach,
    }


if __name__ == '__main__':
    
    paths = InputPaths(
        survey_path='survey.txt',
        flow_table_path='Flow data ell.csv',
        scallop_path='Survey_and_scallop_data.csv',
        output_dir='outputs'
    )

    run_full_model(paths=paths, show_plots=False, validation_path='validation_points.csv')
