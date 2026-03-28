from hitran_processing import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import wofz
import sys, os
from hapi import db_begin, partitionSum
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import pandas as pd



cd = os.getcwd()
path_to_sample = '/Users/ward/Documents/GitHub/Ward-github/Measurement strategies for planetary science missions/AE4895-Measurement_strategies/FTIR/sample B.csv'


#readfile, skip row 0,1, delimiter is ';', decimal seperator is ',', unpack into np_exp and T_exp

df = pd.read_csv(path_to_sample, delimiter=';', decimal=',', skiprows=2)
np_exp = df.iloc[:, 0].values
T_exp = df.iloc[:, 1].values


def main():
    # spectral regions of interest
    REGIONS = {
        ' 600-4000cm-1': (600, 4000),
    }
    NU_MIN_GLOBAL = 600
    NU_MAX_GLOBAL = 4000

    RESOLUTIONS = [100_000, 10_000, 1_000, 200] # set wrt instruments
    RES_LABELS  = ['R = 100,000', 'R = 10,000', 'R = 1,000', 'R = 200']
    RES_COLORS  = ['#38bdf8', '#34d399', '#fbbf24', '#f87171']

    GRID_SPACING = 0.005   # cm⁻¹ — 6× Doppler FWHM at 100K; sufficient for R≤100000

    MOL_IDS   = [MOLEC_H20]
  
    MOL_INDEXES = {MOLEC_CH4: 61, MOLEC_CO2: 21, MOLEC_NH3: 111, MOLEC_CH3_OH: 391, MOLEC_H20: 11}
    MOL_NAMES = {MOLEC_CH4: 'CH₄',MOLEC_CO2: 'CO₂', MOLEC_NH3: 'NH₃',MOLEC_CH3_OH: 'CH₃OH', MOLEC_H20: 'H₂O'}
    MOL_COLS  = {MOLEC_CH4: '#22c55e', MOLEC_CO2: "#833bf6", MOLEC_NH3: '#fbbf24', MOLEC_CH3_OH: '#e879f9', MOLEC_H20: '#38bdf8'}

    # ── Parse ──────────────────────────────────────────────────────────────
    #print(f"\nParsing {parfile} ...")
    # ── Build cross-sections per region per molecule ───────────────────────
    xsec_data = {}   # {mol_id: {region_label: (nu_grid, xsec_raw)}}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    compounds_dir = os.path.join(base_dir, 'Compounds_2e3-4e3cm-1')

    # Build cross-section for H2O only
    mol_id = MOLEC_H20
    xsec_data[mol_id] = {}
    filepath_ind = os.path.join(compounds_dir, '11_600-4000.par')
    lines = get_HITRAN_file_individual(filepath_ind, NU_MIN_GLOBAL, NU_MAX_GLOBAL)
    if not lines:
        print(f"WARNING: no lines found for H2O")
    else:
        print(f"\nBuilding cross-section for H2O ({len(lines)} lines)...")
        
        for reg_label, (nu_lo, nu_hi) in REGIONS.items():
            # filter lines to region + small buffer for wing contributions
            buf = 30.0
            reg_lines = [(nu, sw, ga, gs, el, na)
                         for (nu, sw, ga, gs, el, na) in lines
                         if nu_lo - buf <= nu <= nu_hi + buf]
            print(f"  {reg_label.split(chr(10))[0]}: {len(reg_lines)} lines")
            T_SIM = 18 + 273.15
            P_SIM = 1.0 
            nu_grid = np.arange(nu_lo, nu_hi + GRID_SPACING, GRID_SPACING)
            xsec    = build_xsec(reg_lines, mol_id, nu_grid, T_SIM, P_SIM)
            xsec_data[mol_id][reg_label] = (nu_grid, xsec)

    #
    plt.style.use('default')

    n_rows = len(RESOLUTIONS)
    n_cols = len(REGIONS)
    reg_labels = list(REGIONS.keys())

    fig = plt.figure(figsize=(18, 14))
    # fig.suptitle(
    #     'CH₄ — Absorption Cross-Section from HITRAN Line Data\n'
    #     f'T = {T_SIM:.0f} K, P ≈ {P_SIM:.0e} atm (Doppler-limited, Enceladus plume conditions)',
    #     fontsize=13, fontweight='bold'
    # )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                        hspace=0.2, wspace=0.15)

    for row, (R, rlabel, rcol) in enumerate(zip(RESOLUTIONS, RES_LABELS, RES_COLORS)):
        for col, reg_label in enumerate(reg_labels):
            ax = fig.add_subplot(gs[row, col])

            nu_lo, nu_hi = REGIONS[reg_label]

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

            if row == 0:
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

    # plt.savefig('hitran_ch4_c2h6_resolution.png', dpi=180, bbox_inches='tight')
    # print("\nSaved to hitran_ch4_c2h6_resolution.png")
    plt.show()
    
    # Return raw cross-section and wavenumber grid for H2O
    mol_id = MOLEC_H20
    reg_label = list(REGIONS.keys())[0]
    if mol_id in xsec_data and reg_label in xsec_data[mol_id]:
        nu_grid, xsec_raw = xsec_data[mol_id][reg_label]
        return xsec_raw, nu_grid
    else:
        return None, None

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cd = os.getcwd()
    xsec_raw, nu_grid = main()

    if xsec_raw is None:
        print("ERROR: Could not build H2O cross-section")
        sys.exit(1)

    # Lab conditions
    L = 0.1  # path length in cm
    T_lab = T_SIM  # K
    P_lab = P_SIM  # atm
    print(f"\nLab conditions: T={T_lab:.0f} K, P={P_lab} atm, L={L} cm")
    N = (P_lab * 101325) / (1.38e-23 * T_lab)  # molecules/cm³

    # Optionally convolve to a lab resolution (e.g., R=1000)
    R_lab = 20  # adjust as needed
    xsec_conv = convolve_resolution(nu_grid, xsec_raw, R_lab)

    # Compute absorption coefficient and transmittance
    k_hr = xsec_conv * N  # cm⁻¹
    tau_hr = k_hr * L
    T_hr = np.exp(-tau_hr)

    # Interpolate model to experimental wavenumber grid (1 cm⁻¹ spacing)
    nu_exp = np.arange(600, 4001, 1)
    interp = interp1d(nu_grid, T_hr, bounds_error=False, fill_value=1.0)
    T_model = interp(nu_exp)
    #overplt lab results:
    
    # np_exp, T_exp = np.loadtxt(path_to_sample, delimiter=',', skiprows=2, unpack=True)
    # Plot comparison
    plt.figure(figsize=(12, 5))
    plt.plot(nu_exp, T_model*100, label="HITRAN Model", lw=2)
    plt.plot(np_exp, T_exp, label="Lab Results", lw=2)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Transmittance")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    # plt.title(f"H₂O Transmittance")
    plt.tight_layout()
    plt.savefig('h2o_lab_comparison.png', dpi=300)