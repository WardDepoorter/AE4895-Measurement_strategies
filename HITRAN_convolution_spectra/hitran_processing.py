"""
HITRAN .par spectral simulator
Reads CH4 (mol 6) and C2H6 (mol 27) from a combined .par file,
computes absorption cross-sections using Voigt profiles,
and plots both molecules at multiple spectral resolutions.

Physical model
--------------
Each line contributes a Voigt profile with:
  - Doppler (Gaussian) width from thermal motion at temperature T
  - Lorentzian width from pressure broadening: γ = γ_air * (p/p_ref) * (T_ref/T)^n_air
Line strength is temperature-corrected from 296 K reference via:
  S(T) = S(296) * Q(296)/Q(T) * exp(-c2*E''/T) / exp(-c2*E''/296)
         * [1 - exp(-c2*ν/T)] / [1 - exp(-c2*ν/296)]

For low-pressure (space) conditions pressure broadening → 0,
so the profile is essentially a Gaussian (Doppler-limited).

HITRAN molecule IDs used: CH4 = 6, C2H6 = 27
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import wofz
import sys, os
from hapi import db_begin, partitionSum
cd = os.getcwd()
db_begin(cd + '/Hapifiles')  # initialize HAPI database

# ── Constants ────────────────────────────────────────────────────────────────
C_LIGHT   = 2.99792458e10   # cm/s
K_BOLTZ   = 1.380649e-23    # J/K
N_AVOG    = 6.02214076e23
C2        = 1.4387769       # hc/k in cm·K  (second radiation constant)

# ── Simulation conditions ────────────────────────────────────────────────────
T_SIM     = 100.0           # K  — Enceladus plume / low-T space
P_SIM     = 1e-6            # atm — effectively zero pressure, Doppler-limited
T_REF     = 296.0           # K  — HITRAN reference temperature
P_REF     = 1.0             # atm

# ── Molecule IDs ─────────────────────────────────────────────────────────────
MOLEC_CH4   = 6             #methane
MOLEC_NH3  = 11             #ammonia
MOLEC_CO2 = 2               #carbon dioxide
MOLEC_CH3_OH    = 39        #methanol
MOLEC_H20 = 1               #water

MOLEC_MASS  = {
    6: 16.043, 
    11: 17.031, 
    2: 44.009, 
    39: 32.042, 
    1: 18.01528}            # g/mol (main isotopologue)

# Approximate partition function ratios Q(296)/Q(T) for temperature correction
# From HITRAN TIPS-2020 tables (Fischer et al. 2003; Gamache et al. 2021)
# Using simple polynomial fits valid 70–350 K

def Q_ratio(molec_id, T):
    """Returns Q(T_ref=296) / Q(T) for line strength scaling."""
    iso = 1  # main isotopologue
    Q_T    = partitionSum(molec_id, iso, T)
    Q_ref  = partitionSum(molec_id, iso, 296.0)
    return Q_ref / Q_T  





# ── Voigt profile ─────────────────────────────────────────────────────────────
def voigt(nu, nu0, sigma_D, gamma_L):
    """
    Voigt profile (normalised to unit area).
    sigma_D : Gaussian 1/e half-width (Doppler)
    gamma_L : Lorentzian HWHM (pressure)
    """
    z = ((nu - nu0) + 1j * gamma_L) / (sigma_D * np.sqrt(2))
    return np.real(wofz(z)) / (sigma_D * np.sqrt(2 * np.pi))

def doppler_sigma(nu0, T, mass_g_mol):
    """Gaussian 1/e half-width for Doppler broadening (cm⁻¹)."""
    mass_kg = mass_g_mol * 1e-3 / N_AVOG
    return (nu0 / C_LIGHT) * np.sqrt(K_BOLTZ * T / mass_kg)

def lorentz_gamma(gamma_air, n_air, T, P):
    """Pressure-broadened Lorentzian HWHM (cm⁻¹)."""
    return gamma_air * (P / P_REF) * (T_REF / T) ** n_air



# ── Line strength temperature correction ─────────────────────────────────────
def scale_strength(sw_296, nu0, elower, molec_id, T):
    """Scale HITRAN line strength from 296 K to temperature T."""
    qr = Q_ratio(molec_id, T)
    boltzmann = np.exp(-C2 * elower * (1/T - 1/T_REF))
    stim_296 = 1.0 - np.exp(-C2 * nu0 / T_REF)
    stim_T   = 1.0 - np.exp(-C2 * nu0 / T)
    # avoid division by zero for very low nu (far-IR)
    if abs(stim_296) < 1e-30:
        return sw_296
    return sw_296 * qr * boltzmann * (stim_T / stim_296)

# ── HITRAN parser ─────────────────────────────────────────────────────────────
def parse_hitran_par(filepath, molec_ids, nu_min, nu_max):
    """
    Parse a HITRAN .par file and return a dict of line lists per molecule.
    Only keeps lines within [nu_min, nu_max].
    molec_ids: list of integer molecule IDs to keep
    Returns: dict {molec_id: list of (nu, sw, gamma_air, gamma_self, elower, n_air)}
    """
    lines_out = {m: [] for m in molec_ids}
    n_total = 0
    n_kept  = {m: 0 for m in molec_ids}

    with open(filepath, 'r') as f:
        for raw in f:
            if len(raw) < 100:
                continue
            try:
                mid = int(raw[0:2])
            except ValueError:
                continue
            if mid not in molec_ids:
                continue
            try:
                nu        = float(raw[3:15])
                if not (nu_min <= nu <= nu_max):
                    continue
                sw        = float(raw[15:25])
                gamma_air = float(raw[35:40])
                gamma_self= float(raw[40:45])
                elower    = float(raw[45:55])
                n_air     = float(raw[55:59])
            except ValueError:
                continue

            lines_out[mid].append((nu, sw, gamma_air, gamma_self, elower, n_air))
            n_kept[mid] += 1
            n_total += 1

    print(f"Parsed {n_total} lines total")
    for m in molec_ids:
        print(f"  Mol {m}: {n_kept[m]} lines in [{nu_min}, {nu_max}] cm⁻¹")
    return lines_out

def get_HITRAN_file_individual(filepath, nu_min, nu_max):
    """
    Read hitran transition files for individual molecules.
    """
    if not isinstance(filepath, (str, os.PathLike)):
        raise TypeError(f"filepath must be a path-like string, got {type(filepath).__name__}: {filepath!r}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HITRAN file not found: {filepath}")

    lines_out = []
    n_total = 0

    with open(filepath, 'r') as f:
        for raw in f:
            if len(raw) < 100:
                continue
            try:
                nu = float(raw[3:15])
                if not (nu_min <= nu <= nu_max):
                    continue
                sw         = float(raw[15:25])
                gamma_air  = float(raw[35:40])
                gamma_self = float(raw[40:45])
                elower     = float(raw[45:55])
                n_air      = float(raw[55:59])
            except ValueError:
                continue

            lines_out.append((nu, sw, gamma_air, gamma_self, elower, n_air))
            n_total += 1

    # print(f"Mol }: {n_total} lines in [{nu_min}, {nu_max}] cm⁻¹")
    return lines_out
    
    
# ── Cross-section grid builder ─────────────────────────────────────────────
def build_xsec(line_list, molec_id, nu_grid, T, P,
               wing_cutoff_cm1=0.8, progress=True):
    """
    Build absorption cross-section on nu_grid (cm⁻¹) from line list.
    wing_cutoff_cm1: ignore line contributions beyond this distance from line centre.
    Returns cross-section array in cm²/molecule.
    """
    mass = MOLEC_MASS.get(molec_id, 20.0)
    xsec = np.zeros(len(nu_grid))
    dnu  = nu_grid[1] - nu_grid[0]
    n    = len(line_list)

    for i, (nu0, sw_296, gamma_air, gamma_self, elower, n_air) in enumerate(line_list):
        if progress and i % 50000 == 0 and i > 0:
            print(f"    Line {i}/{n}...", flush=True)

        sw_T    = scale_strength(sw_296, nu0, elower, molec_id, T)
        if sw_T < 1e-45:   # skip negligible lines
            continue

        sig_D   = doppler_sigma(nu0, T, mass)
        gam_L   = lorentz_gamma(gamma_air, n_air, T, P)

        # index range within cutoff
        i_lo = max(0, int((nu0 - wing_cutoff_cm1 - nu_grid[0]) / dnu))
        i_hi = min(len(nu_grid), int((nu0 + wing_cutoff_cm1 - nu_grid[0]) / dnu) + 1)
        if i_lo >= i_hi:
            continue

        sub = nu_grid[i_lo:i_hi]
        xsec[i_lo:i_hi] += sw_T * voigt(sub, nu0, sig_D, gam_L)

    return xsec

# ── Instrument resolution convolution ─────────────────────────────────────
def convolve_resolution(nu_grid, xsec, R):
    """
    Convolve with Gaussian instrument line shape of resolving power R.
    FWHM_inst = nu_centre / R
    """
    from scipy.ndimage import gaussian_filter1d
    nu_c   = 0.5 * (nu_grid[0] + nu_grid[-1])
    fwhm   = nu_c / R
    sigma_wn = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dnu    = nu_grid[1] - nu_grid[0]
    sigma_px = sigma_wn / dnu
    if sigma_px < 0.3:
        return xsec
    return gaussian_filter1d(xsec, sigma=sigma_px)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # spectral regions of interest
    REGIONS = {
        ' 2000-4000cm-1': (2000, 4000),
    }
    NU_MIN_GLOBAL = 2000
    NU_MAX_GLOBAL = 4000

    RESOLUTIONS = [100_000, 800, 350, 200] # set wrt instruments
    RES_LABELS  = ['No smoothing(R = 100,000)', 'Perkin-Elmer 100(R = 800)', 'MISE(R = 350)', 'VIMS(R =200)']
    RES_COLORS  = ['#38bdf8', '#34d399', '#fbbf24', '#f87171']

    GRID_SPACING = 0.005   # cm⁻¹ — 6× Doppler FWHM at 100K; sufficient for R≤100000

    MOL_IDS   = [MOLEC_CH4, MOLEC_CO2, MOLEC_NH3, MOLEC_CH3_OH, MOLEC_H20]
  
    MOL_INDEXES = {MOLEC_CH4: 61, MOLEC_CO2: 21, MOLEC_NH3: 111, MOLEC_CH3_OH: 391, MOLEC_H20: 11}
    MOL_NAMES = {MOLEC_CH4: 'CH₄',MOLEC_CO2: 'CO₂', MOLEC_NH3: 'NH₃',MOLEC_CH3_OH: 'CH₃OH', MOLEC_H20: 'H₂O'}
    MOL_COLS  = {MOLEC_CH4: '#22c55e', MOLEC_CO2: "#833bf6", MOLEC_NH3: '#fbbf24', MOLEC_CH3_OH: '#e879f9', MOLEC_H20: '#38bdf8'}

    # ── Parse ──────────────────────────────────────────────────────────────
    #print(f"\nParsing {parfile} ...")
    # ── Build cross-sections per region per molecule ───────────────────────
    xsec_data = {}   # {mol_id: {region_label: (nu_grid, xsec_raw)}}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    compounds_dir = os.path.join(base_dir, 'Compounds_2e3-4e3cm-1')

    for mol_id in MOL_IDS:
        xsec_data[mol_id] = {}
        filepath_ind = os.path.join(compounds_dir, f'{MOL_INDEXES[mol_id]}.par')
        lines = get_HITRAN_file_individual(filepath_ind, NU_MIN_GLOBAL, NU_MAX_GLOBAL)
        if not lines:
            print(f"WARNING: no lines found for molecule {mol_id}")
            continue
        print(f"\nBuilding cross-section for {MOL_NAMES[mol_id]} ({len(lines)} lines)...")

        for reg_label, (nu_lo, nu_hi) in REGIONS.items():
            # filter lines to region + small buffer for wing contributions
            buf = 30.0
            reg_lines = [(nu, sw, ga, gs, el, na)
                         for (nu, sw, ga, gs, el, na) in lines
                         if nu_lo - buf <= nu <= nu_hi + buf]
            print(f"  {reg_label.split(chr(10))[0]}: {len(reg_lines)} lines")

            nu_grid = np.arange(nu_lo, nu_hi + GRID_SPACING, GRID_SPACING)
            xsec    = build_xsec(reg_lines, mol_id, nu_grid, T_SIM, P_SIM)
            xsec_data[mol_id][reg_label] = (nu_grid, xsec)

    # # ── Plot ───────────────────────────────────────────────────────────────
    # n_rows = len(RESOLUTIONS)
    # n_cols = len(REGIONS)
    # reg_labels = list(REGIONS.keys())

    # fig = plt.figure(figsize=(18, 14), facecolor='#0f1117')
    # fig.suptitle(
    #     'CH₄  —  Absorption Cross-Section from HITRAN Line Data\n'
    #     f'T = {T_SIM:.0f} K, P ≈ {P_SIM:.0e} atm  (Doppler-limited, Enceladus plume conditions)',
    #     color='white', fontsize=13, fontweight='bold', y=0.99
    # )

    # gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
    #                        hspace=0.07, wspace=0.06,
    #                        left=0.07, right=0.97, top=0.93, bottom=0.07)

    # for row, (R, rlabel, rcol) in enumerate(zip(RESOLUTIONS, RES_LABELS, RES_COLORS)):
    #     for col, reg_label in enumerate(reg_labels):
    #         ax = fig.add_subplot(gs[row, col])
    #         ax.set_facecolor('#0f1117')
    #         ax.spines[:].set_color('#1e293b')
    #         ax.tick_params(colors='#64748b', labelsize=8)

    #         nu_lo, nu_hi = REGIONS[reg_label]
    #         plotted_any = False

    #         peak_vals = {}
    #         convolved = {}

    #         for mol_id in MOL_IDS:
    #             if reg_label not in xsec_data[mol_id]:
    #                 continue
    #             nu_grid, xsec_raw = xsec_data[mol_id][reg_label]
    #             xsec_conv = convolve_resolution(nu_grid, xsec_raw, R)
    #             convolved[mol_id] = (nu_grid, xsec_conv)
    #             peak_vals[mol_id] = xsec_conv.max() if xsec_conv.max() > 0 else 1.0

    #         # normalise to global max across both molecules in this panel
    #         global_max = max(peak_vals.values()) if peak_vals else 1.0

    #         for mol_id in MOL_IDS:
    #             if mol_id not in convolved:
    #                 continue
    #             nu_grid, xc = convolved[mol_id]
    #             xc_norm = xc / global_max
    #             c = MOL_COLS[mol_id]
    #             ax.fill_between(nu_grid, xc_norm, alpha=0.12, color=c)
    #             ax.plot(nu_grid, xc_norm, color=c, lw=0.9,
    #                     label=MOL_NAMES[mol_id])
    #             plotted_any = True

    #         ax.set_xlim(nu_lo, nu_hi)
    #         ax.set_ylim(-0.04, 1.18)

    #         # Resolution badge (left)
    #         ax.text(0.02, 0.90, rlabel,
    #                 transform=ax.transAxes, color=rcol, fontsize=9,
    #                 fontweight='bold',
    #                 bbox=dict(boxstyle='round,pad=0.3', fc='#0f1117',
    #                           ec=rcol, alpha=0.9))

    #         # # Distinguishability metric (right)
    #         # if len(convolved) == 2:
    #         #     a = convolved[MOLEC_CH4][1]   / global_max
    #         #     b = convolved[MOLEC_C2H6][1]  / global_max
    #         #     overlap = (np.trapezoid(np.minimum(a, b)) /
    #         #                max(np.trapezoid(np.maximum(a, b)), 1e-30))
    #         #     if overlap < 0.40:
    #         #         badge, bc = '✓ Distinguishable', '#22c55e'
    #         #     elif overlap < 0.70:
    #         #         badge, bc = '~ Marginal',        '#eab308'
    #         #     else:
    #         #         badge, bc = '✗ Merged',          '#ef4444'
    #         #     ax.text(0.98, 0.90, badge, transform=ax.transAxes,
    #         #             color=bc, fontsize=8, ha='right',
    #         #             bbox=dict(boxstyle='round,pad=0.3', fc='#0f1117',
    #         #                       ec=bc, alpha=0.85))

    #         # Column title (top row only)
    #         if row == 0:
    #             ax.set_title(reg_label, color='white', fontsize=11, pad=6)
    #             ax.legend(facecolor='#1e293b', edgecolor='#334155',
    #                       labelcolor='white', fontsize=9,
    #                       loc='upper right' if col==0 else 'upper left')

    #         # x-axis label (bottom row only)
    #         if row == n_rows - 1:
    #             ax.set_xlabel('Wavenumber (cm⁻¹)', color='#94a3b8', fontsize=9)
    #         else:
    #             ax.set_xticklabels([])

    #         # y-axis label (left column only)
    #         if col == 0:
    #             ax.set_ylabel('Norm. Cross-Section', color='#94a3b8', fontsize=8)
    #         else:
    #             ax.set_yticklabels([])

    # # Footer
    # fig.text(0.5, 0.005,
    #          f'HITRAN line data; Voigt profiles; T={T_SIM:.0f}K, P={P_SIM:.0e}atm. '
    #          'Instrument LSF: Gaussian with FWHM = ν_centre/R.',
    #          ha='center', color='#475569', fontsize=7.5)


    # plt.savefig('hitran_ch4_c2h6_resolution.png', dpi=180, bbox_inches='tight', facecolor='#0f1117')
    # print(f"\nSaved to hitran_ch4_c2h6_resolution.png")
# ── Plot ─────────────────────────────────────────────
    plt.style.use('default')

    n_rows = 2
    n_cols = 2
    reg_labels = list(REGIONS.keys())

    fig = plt.figure(figsize=(18, 14))
    # fig.suptitle(
    #     'CH₄ — Absorption Cross-Section from HITRAN Line Data\n'
    #     f'T = {T_SIM:.0f} K, P ≈ {P_SIM:.0e} atm (Doppler-limited, Enceladus plume conditions)',
    #     fontsize=13, fontweight='bold'
    # )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                        hspace=0.2, wspace=0.15)

    reg_label = reg_labels[0]
    nu_lo, nu_hi = REGIONS[reg_label]

    for idx, (R, rlabel, rcol) in enumerate(zip(RESOLUTIONS, RES_LABELS, RES_COLORS)):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        peak_vals = {}
        convolved = {}

        for mol_id in MOL_IDS:
            if reg_label not in xsec_data[mol_id]:
                continue
            nu_grid, xsec_raw = xsec_data[mol_id][reg_label]
            xsec_conv = convolve_resolution(nu_grid, xsec_raw, R)
            convolved[mol_id] = (nu_grid, xsec_conv)
            peak_vals[mol_id] = xsec_conv.max() if xsec_conv.max() > 0 else 1.0

        global_max = max(peak_vals.values()) if peak_vals else 1.0

        for mol_id in MOL_IDS:
            if mol_id not in convolved:
                continue
            nu_grid, xc = convolved[mol_id]
            xc_norm = xc / global_max
            c = MOL_COLS[mol_id]

            ax.fill_between(nu_grid, xc_norm, alpha=0.2, color=c)
            ax.plot(nu_grid, xc_norm, lw=1.0, color=c, label=MOL_NAMES[mol_id])

        ax.set_xlim(nu_lo, nu_hi)
        ax.set_ylim(-0.04, 1.18)

        # Resolution label
        ax.text(0.02, 0.90, rlabel, transform=ax.transAxes, fontsize=9)

        if idx == 0:
            ax.set_title(reg_label, fontsize=11)
            ax.legend(fontsize=9)

        if row == n_rows - 1:
            ax.set_xlabel('Wavenumber (cm⁻¹)')
        else:
            ax.set_xticklabels([])

        if col == 0:
            ax.set_ylabel('absorption(norm.)')
        else:
            ax.set_yticklabels([])

    # fig.text(0.5, 0.01,
    #         f'HITRAN line data; Voigt profiles; T={T_SIM:.0f}K, P={P_SIM:.0e}atm. '
    #         'Instrument LSF: Gaussian with FWHM = ν_centre/R.',
    #         ha='center', fontsize=8)
    plt.show()
    plt.savefig('hitran_ch4_c2h6_resolution.png', dpi=300, bbox_inches='tight')
    print("\nSaved to hitran_ch4_c2h6_resolution.png")
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cd = os.getcwd()
    main()   # ← put your filename/path here