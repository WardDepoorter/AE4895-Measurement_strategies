"""
hitran_instrument_comparison.py
================================
Plots CH4 vs C2H6 absorption cross-sections at the spectral resolution of
four real instruments:
  - Perkin-Elmer Spectrum 100 FTIR  (4 cm-1 default, 0.5 cm-1 best)
  - Cassini VIMS IR                  (16.6 nm/channel)
  - Europa Clipper MISE              (10 nm/channel)

Uses a HITRAN .par file (160-char fixed-width HITRAN2004 format) as input.

Usage
-----
    python3 hitran_instrument_comparison.py <path_to_combined.par>

The .par file should contain both CH4 (molecule ID 6) and C2H6 (molecule ID 27)
in the 600-3200 cm-1 range. Download from hitran.org/lbl/.

Dependencies
------------
    numpy, scipy, matplotlib

Physical model
--------------
- Voigt line profiles: Doppler (Gaussian) width from thermal motion at T,
  Lorentzian width from pressure broadening gamma_air*(P/P_ref)*(T_ref/T)^n_air
- Line strengths temperature-corrected from HITRAN 296 K reference
- Low-pressure limit (Enceladus plume): effectively Doppler-limited Gaussians
- Grid spacing: 0.005 cm-1 (sufficient for R <= 100,000 at these conditions)
- Wing cutoff: 0.8 cm-1 (>>30x Doppler FWHM at 100 K)

References
----------
- HITRAN2004 format: Rothman et al., JQSRT 96, 139 (2005)
- CH4 line data: Brown et al., JQSRT 130, 201 (2013)
- C2H6 line data: Vander Auwera et al., ApJ Suppl 173, 522 (2007)
- Cassini VIMS: Brown et al., Space Sci Rev 115, 111 (2004)
- Europa Clipper MISE: Blaney et al., Space Sci Rev 219, 32 (2023)
- PE Spectrum 100: PerkinElmer product specifications
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import wofz
from scipy.ndimage import gaussian_filter1d

# ── Simulation conditions ─────────────────────────────────────────────────────
T_SIM   = 100.0     # K   — Enceladus plume temperature
P_SIM   = 1e-6      # atm — effectively zero pressure (Doppler-limited)
T_REF   = 296.0     # K   — HITRAN reference temperature
P_REF   = 1.0       # atm — HITRAN reference pressure

# ── Physical constants ────────────────────────────────────────────────────────
C_LIGHT = 2.99792458e10   # cm/s
K_BOLTZ = 1.380649e-23    # J/K
N_AVOG  = 6.02214076e23
C2      = 1.4387769       # hc/k in cm·K (second radiation constant)

# ── Molecule definitions ──────────────────────────────────────────────────────
MOLEC_CH4  = 6
MOLEC_C2H6 = 27
MOLEC_MASS = {MOLEC_CH4: 16.043, MOLEC_C2H6: 30.069}   # g/mol
MOL_NAMES  = {MOLEC_CH4: 'CH₄',  MOLEC_C2H6: 'C₂H₆'}
MOL_COLORS = {MOLEC_CH4: '#60a5fa', MOLEC_C2H6: '#fb923c'}

# ── Spectral regions ──────────────────────────────────────────────────────────
REGIONS = {
    'C–H Stretch\n(2800–3200 cm⁻¹)': (2800, 3200),
    'Fingerprint\n(600–1600 cm⁻¹)':  (600,  1600),
}
# Representative centre wavenumber per region (used to compute R for nm-based instruments)
NU_CENTRE = {
    'C–H Stretch\n(2800–3200 cm⁻¹)': 3000.0,
    'Fingerprint\n(600–1600 cm⁻¹)':  1100.0,
}

# ── Instrument definitions ────────────────────────────────────────────────────
# Each entry: (display_label, plot_colour, spec_type, spec_value)
# spec_type 'cm1' : fixed spectral resolution in cm-1 (FTIR instruments)
# spec_type 'nm'  : fixed channel width in nm (grating/prism imagers)
INSTRUMENTS = [
    ('PE-100  (4 cm⁻¹, default)', '#a78bfa', 'cm1', 4.0),
    ('PE-100  (0.5 cm⁻¹, best)',  '#38bdf8', 'cm1', 0.5),
    ('Cassini VIMS IR (16.6 nm)', '#fb923c', 'nm',  16.6),
    ('MISE  (10 nm)',             '#34d399', 'nm',  10.0),
]

GRID_SPACING   = 0.005   # cm-1
WING_CUTOFF    = 0.8     # cm-1 — line profile wing truncation


# ─────────────────────────────────────────────────────────────────────────────
# Physics functions
# ─────────────────────────────────────────────────────────────────────────────

def partition_function_ratio(molec_id, T):
    """
    Returns Q(T_ref=296 K) / Q(T) for line strength temperature scaling.
    Approximated as power-law fits to HITRAN TIPS-2020 tables.
    """
    if molec_id == MOLEC_CH4:
        return (T / T_REF) ** 1.5    # symmetric top approximation
    elif molec_id == MOLEC_C2H6:
        return (T / T_REF) ** 1.75   # slightly higher due to internal rotation
    return 1.0


def voigt_profile(nu, nu0, sigma_D, gamma_L):
    """
    Normalised Voigt profile (unit area) at wavenumbers nu.
    sigma_D : Gaussian 1/e half-width (Doppler broadening), cm-1
    gamma_L : Lorentzian HWHM (pressure broadening), cm-1
    """
    z = ((nu - nu0) + 1j * gamma_L) / (sigma_D * np.sqrt(2))
    return np.real(wofz(z)) / (sigma_D * np.sqrt(2 * np.pi))


def doppler_sigma(nu0, T, mass_g_mol):
    """Gaussian 1/e half-width for Doppler broadening (cm-1)."""
    mass_kg = mass_g_mol * 1e-3 / N_AVOG
    return (nu0 / C_LIGHT) * np.sqrt(K_BOLTZ * T / mass_kg)


def pressure_gamma(gamma_air, n_air, T, P):
    """Lorentzian HWHM from air-pressure broadening (cm-1)."""
    return gamma_air * (P / P_REF) * (T_REF / T) ** n_air


def temperature_correct_strength(sw_296, nu0, elower, molec_id, T):
    """
    Scale HITRAN line strength from 296 K to temperature T.
    Follows the standard HITRAN temperature correction formula.
    """
    qr       = partition_function_ratio(molec_id, T)
    boltzmann = np.exp(-C2 * elower * (1.0/T - 1.0/T_REF))
    stim_296  = 1.0 - np.exp(-C2 * nu0 / T_REF)
    stim_T    = 1.0 - np.exp(-C2 * nu0 / T)
    if abs(stim_296) < 1e-30:
        return sw_296
    return sw_296 * qr * boltzmann * (stim_T / stim_296)


def instrument_R(spec_type, spec_value, nu_centre):
    """
    Compute resolving power R = nu / delta_nu at nu_centre.
    spec_type 'cm1': delta_nu is fixed in cm-1
    spec_type 'nm' : delta_lambda is fixed in nm
    """
    if spec_type == 'cm1':
        return nu_centre / spec_value
    else:
        lam_um  = 1e4 / nu_centre       # wavenumber → wavelength in um
        dlam_um = spec_value * 1e-3      # nm → um
        return lam_um / dlam_um


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def parse_hitran_par(filepath, molec_ids, nu_min, nu_max):
    """
    Parse a HITRAN .par file (HITRAN2004 160-char fixed-width format).

    Returns dict {molec_id: [(nu, sw, gamma_air, gamma_self, elower, n_air), ...]}
    Only lines within [nu_min, nu_max] are retained.
    """
    out = {m: [] for m in molec_ids}
    with open(filepath, 'r') as f:
        for line in f:
            if len(line) < 100:
                continue
            try:
                mid = int(line[0:2])
                if mid not in molec_ids:
                    continue
                nu        = float(line[3:15])
                if not (nu_min <= nu <= nu_max):
                    continue
                sw        = float(line[15:25])
                gamma_air = float(line[35:40])
                gamma_self= float(line[40:45])
                elower    = float(line[45:55])
                n_air     = float(line[55:59])
            except ValueError:
                continue
            out[mid].append((nu, sw, gamma_air, gamma_self, elower, n_air))

    for m in molec_ids:
        print(f'  Mol {m:2d} ({MOL_NAMES.get(m,"?")}): {len(out[m]):>7,} lines '
              f'in [{nu_min:.0f}, {nu_max:.0f}] cm-1')
    return out


def build_cross_section(line_list, molec_id, nu_grid, T, P):
    """
    Compute absorption cross-section on nu_grid from line_list.
    Returns array in cm2/molecule (not normalised).
    """
    mass = MOLEC_MASS.get(molec_id, 20.0)
    xsec = np.zeros(len(nu_grid))
    dnu  = nu_grid[1] - nu_grid[0]

    for nu0, sw_296, gamma_air, _, elower, n_air in line_list:
        sw_T = temperature_correct_strength(sw_296, nu0, elower, molec_id, T)
        if sw_T < 1e-45:
            continue
        sig_D = doppler_sigma(nu0, T, mass)
        gam_L = pressure_gamma(gamma_air, n_air, T, P)

        i_lo = max(0, int((nu0 - WING_CUTOFF - nu_grid[0]) / dnu))
        i_hi = min(len(nu_grid), int((nu0 + WING_CUTOFF - nu_grid[0]) / dnu) + 2)
        if i_lo >= i_hi:
            continue

        sub = nu_grid[i_lo:i_hi]
        xsec[i_lo:i_hi] += sw_T * voigt_profile(sub, nu0, sig_D, gam_L)

    return xsec


def convolve_to_resolution(nu_grid, xsec, R):
    """
    Convolve cross-section with Gaussian instrument line shape of resolving power R.
    FWHM_instrument = nu_centre / R
    """
    nu_c     = 0.5 * (nu_grid[0] + nu_grid[-1])
    fwhm     = nu_c / R
    sigma_wn = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
    sigma_px = sigma_wn / (nu_grid[1] - nu_grid[0])
    if sigma_px < 0.3:
        return xsec
    return gaussian_filter1d(xsec, sigma=sigma_px)


def overlap_metric(a, b):
    """
    Spectral overlap metric: integral of min(a,b) / integral of max(a,b).
    0 = no overlap (perfectly distinguishable), 1 = identical.
    """
    denom = np.trapezoid(np.maximum(a, b))
    if denom < 1e-30:
        return 1.0
    return np.trapezoid(np.minimum(a, b)) / denom


def distinguishability_badge(overlap):
    if overlap < 0.40:
        return '✓ Distinguishable', '#22c55e'
    elif overlap < 0.70:
        return '~ Marginal', '#eab308'
    else:
        return '✗ Merged', '#ef4444'


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(parfile):

    MOL_IDS    = [MOLEC_CH4, MOLEC_C2H6]
    reg_labels = list(REGIONS.keys())

    # ── Parse ──────────────────────────────────────────────────────────────
    print(f'\nParsing {parfile} ...')
    line_data = parse_hitran_par(parfile, MOL_IDS, 600, 3200)

    # ── Build raw cross-sections per region per molecule ───────────────────
    print('\nBuilding cross-sections...')
    xsec_raw = {}
    for mid in MOL_IDS:
        xsec_raw[mid] = {}
        lines = line_data[mid]
        for reg, (lo, hi) in REGIONS.items():
            buf = 30.0
            reg_lines = [(nu,sw,ga,gs,el,na) for nu,sw,ga,gs,el,na in lines
                         if lo - buf <= nu <= hi + buf]
            nu_grid = np.arange(lo, hi + GRID_SPACING, GRID_SPACING)
            xsec    = build_cross_section(reg_lines, mid, nu_grid, T_SIM, P_SIM)
            xsec_raw[mid][reg] = (nu_grid, xsec)
            print(f'  {MOL_NAMES[mid]:5s}  {reg.split(chr(10))[0]:20s}: '
                  f'{len(reg_lines):>6,} lines, grid {len(nu_grid):>6,} pts')

    # ── Plot ───────────────────────────────────────────────────────────────
    n_rows = len(INSTRUMENTS)
    n_cols = len(REGIONS)

    fig = plt.figure(figsize=(18, 13), facecolor='#0f1117')
    fig.suptitle(
        'CH₄ vs C₂H₆  —  Instrument Resolution Comparison\n'
        f'HITRAN line data · T = {T_SIM:.0f} K · P ≈ {P_SIM:.0e} atm '
        '(Enceladus plume conditions)',
        color='white', fontsize=13, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.07, wspace=0.06,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    for row, (inst_label, inst_col, inst_type, inst_val) in enumerate(INSTRUMENTS):
        for col, reg in enumerate(reg_labels):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#0f1117')
            ax.spines[:].set_color('#1e293b')
            ax.tick_params(colors='#64748b', labelsize=8)

            nu_lo, nu_hi = REGIONS[reg]
            nu_c         = NU_CENTRE[reg]
            R            = instrument_R(inst_type, inst_val, nu_c)

            # Convolve and normalise
            convolved = {}
            peak_vals = {}
            for mid in MOL_IDS:
                grid, xraw = xsec_raw[mid][reg]
                xc = convolve_to_resolution(grid, xraw, R)
                convolved[mid] = (grid, xc)
                peak_vals[mid] = xc.max() if xc.max() > 0 else 1.0

            global_max = max(peak_vals.values())

            for mid in MOL_IDS:
                grid, xc = convolved[mid]
                xn = xc / global_max
                c  = MOL_COLORS[mid]
                ax.fill_between(grid, xn, alpha=0.12, color=c)
                ax.plot(grid, xn, color=c, lw=0.9, label=MOL_NAMES[mid])

            ax.set_xlim(nu_lo, nu_hi)
            ax.set_ylim(-0.04, 1.22)

            # Instrument label + R value
            ax.text(0.02, 0.90,
                    f'{inst_label}\nR ≈ {R:.0f}  at {nu_c:.0f} cm⁻¹',
                    transform=ax.transAxes, color=inst_col,
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#0f1117',
                              ec=inst_col, alpha=0.9))

            # Distinguishability badge
            a = convolved[MOLEC_CH4][1]  / global_max
            b = convolved[MOLEC_C2H6][1] / global_max
            ov = overlap_metric(a, b)
            badge, bc = distinguishability_badge(ov)
            ax.text(0.98, 0.90, badge,
                    transform=ax.transAxes, color=bc, fontsize=8, ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#0f1117',
                              ec=bc, alpha=0.85))

            # Column titles (top row only)
            if row == 0:
                ax.set_title(reg, color='white', fontsize=11, pad=6)
                ax.legend(facecolor='#1e293b', edgecolor='#334155',
                          labelcolor='white', fontsize=9,
                          loc='upper right' if col == 0 else 'upper left')

            # Axis labels
            if row < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Wavenumber (cm⁻¹)', color='#94a3b8', fontsize=9)

            if col == 0:
                ax.set_ylabel('Norm. Cross-Section', color='#94a3b8', fontsize=8)
            else:
                ax.set_yticklabels([])

    fig.text(
        0.5, 0.005,
        'HITRAN line data · Voigt profiles · T=100K, P≈0atm · '
        'VIMS: 16.6 nm/channel (0.85–5.1 µm) · MISE: 10 nm/channel (0.8–5 µm) · '
        'PE-100: fixed Δν in cm⁻¹ (R varies with wavenumber)',
        ha='center', color='#475569', fontsize=7.5
    )

    outpath = 'hitran_instrument_comparison.png'
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor='#0f1117')
    print(f'\nSaved to {outpath}')


if __name__ == '__main__':
    cd = os.getcwd()
    main(cd + '/Compounds_2e3-4e3cm-1/69c12293_original.par')   # ← put your filename/path here
