"""
VIMS Raw Cube SNR Analysis
--------------------------
Reads a raw Cassini VIMS .qub file directly (no ISIS3 needed),
extracts spectra and computes SNR using spatial method on a uniform patch.

Usage:
    Edit CUBE_PATH and run: python vims_snr_analysis.py

Label key parameters (from v1487298702_1.qub):
    CORE_ITEMS     = (64, 352, 48)   -> (n_samples, n_bands, n_lines)
    SUFFIX_ITEMS   = (1, 4, 0)       -> 1 sample suffix, 4 band suffixes
    CORE_ITEM_BYTES = 2 (SUN_INTEGER, big-endian)
    SUFFIX_BYTES    = 4
    ^QUBE           = 47             -> data starts at record 47
    RECORD_BYTES    = 512
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
current_dir = os.getcwd()
# ---- CONFIGURATION ----
CUBE_PATH = current_dir + '/raw_flyby_data/'+'v1487298702_1.qub'   # path to your raw .qub file
ALTITUDE_KM = 115                 # approximate flyby altitude for this cube
                                  # update per cube from OPUS/Tour Atlas

# SNR windows
REFERENCE_WINDOW = (2.0, 2.2)    # featureless water ice region
ORGANIC_WINDOW   = (3.0, 3.5)    # C-H stretch organics window

# Spatial patch for SNR (avoid tiger stripes - check image preview first)
PATCH_ROWS = slice(20, 30)
PATCH_COLS = slice(25, 40)


# ---- READ RAW CUBE ----
def read_vims_qub(filepath):
    """
    Parse a raw VIMS .qub PDS file directly into a numpy array.
    Returns cube (lines, samples, bands) in raw DN, and wavelength array.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    # Parse label
    label_end = raw.find(b'END\r\n')
    label_text = raw[:label_end + 5].decode('ascii', errors='ignore')

    # Fixed parameters read directly from label inspection
    n_samples    = 64
    n_bands      = 352
    n_lines      = 48
    record_bytes = 512
    qube_record  = 47
    data_offset  = (qube_record - 1) * record_bytes

    # Binary layout per line:
    # For each BAND: n_samples * 2 bytes (core) + 1 * 4 bytes (sample suffix)
    # After all bands: 4 * (n_samples + 1) * 4 bytes (band suffixes)
    bytes_per_band = n_samples * 2 + 1 * 4           # 132 bytes
    bytes_band_sfx = 4 * (n_samples + 1) * 4         # 1040 bytes
    bytes_per_line = n_bands * bytes_per_band + bytes_band_sfx

    cube = np.zeros((n_lines, n_bands, n_samples), dtype=np.int16)
    for line in range(n_lines):
        line_start = data_offset + line * bytes_per_line
        for band in range(n_bands):
            band_offset = line_start + band * bytes_per_band
            vals = np.frombuffer(
                raw[band_offset:band_offset + n_samples * 2], dtype='>i2'
            )
            cube[line, band, :] = vals

    # Reorder to (lines, samples, bands) - matches pyvims convention
    cube = np.transpose(cube, (0, 2, 1))

    # Parse wavelengths from label
    wav_match = re.search(
        r'BAND_BIN_CENTER\s*=\s*\(([^)]+)\)', label_text, re.DOTALL
    )
    wav_str = wav_match.group(1).replace('\r','').replace('\n','').replace(' ','')
    wav = np.array([float(x) for x in wav_str.split(',')])

    # Mask null/saturation values
    cube = cube.astype(float)
    cube[cube < -390] = np.nan

    return cube, wav


# ---- COMPUTE SNR ----
def compute_spatial_snr(cube, wav, patch_rows, patch_cols, window):
    """
    Compute SNR from spatial variability across a uniform patch at a given
    spectral window. Signal = mean DN across patch, Noise = std across patch.
    This is done per band then averaged across the window.
    """
    wmin, wmax = window
    band_mask = (wav >= wmin) & (wav <= wmax)

    patch = cube[patch_rows, patch_cols, :]   # shape (patch_h, patch_w, bands)
    patch_window = patch[:, :, band_mask]     # restrict to window

    # Per-band SNR across patch pixels
    signal_per_band = np.nanmean(patch_window, axis=(0, 1))
    noise_per_band  = np.nanstd(patch_window,  axis=(0, 1))

    snr_per_band = np.where(noise_per_band > 0,
                            signal_per_band / noise_per_band, np.nan)

    return float(np.nanmedian(snr_per_band)), signal_per_band, noise_per_band, wav[band_mask]


# ---- MAIN ----
if __name__ == '__main__':

    print(f'Loading: {CUBE_PATH}')
    cube, wav = read_vims_qub(CUBE_PATH)
    print(f'Cube shape (lines, samples, bands): {cube.shape}')
    print(f'Wavelength range: {wav.min():.3f} - {wav.max():.3f} µm')

    # --- SNR in reference window ---
    snr_ref, sig_ref, noi_ref, wav_ref = compute_spatial_snr(
        cube, wav, PATCH_ROWS, PATCH_COLS, REFERENCE_WINDOW
    )
    print(f'\nSNR in {REFERENCE_WINDOW} µm reference window: {snr_ref:.1f}')

    # --- SNR in organic window ---
    snr_org, sig_org, noi_org, wav_org = compute_spatial_snr(
        cube, wav, PATCH_ROWS, PATCH_COLS, ORGANIC_WINDOW
    )
    print(f'SNR in {ORGANIC_WINDOW} µm organic window:    {snr_org:.1f}')
    print(f'Altitude: {ALTITUDE_KM} km')

    # --- Plot 1: example spectrum from centre pixel ---
    fig, ax = plt.subplots(figsize=(11, 4))
    spec = cube[24, 32, :]
    ax.plot(wav, spec, lw=0.8, color='steelblue', label='Centre pixel (24,32)')
    ax.axvspan(*REFERENCE_WINDOW, alpha=0.15, color='orange', label='Reference window')
    ax.axvspan(*ORGANIC_WINDOW,   alpha=0.15, color='green',  label='Organic window (C-H)')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Raw DN')
    ax.set_title(f'Raw VIMS spectrum — {CUBE_PATH}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('raw_spectrum.png', dpi=150)
    print('Saved: raw_spectrum.png')

    # --- Plot 2: SNR per band in each window ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(wav_ref, sig_ref / noi_ref, color='orange')
    axes[0].axhline(snr_ref, ls='--', color='black', label=f'Median SNR={snr_ref:.1f}')
    axes[0].set_title(f'SNR per band — reference window {REFERENCE_WINDOW} µm')
    axes[0].set_xlabel('Wavelength (µm)')
    axes[0].set_ylabel('SNR')
    axes[0].legend()

    axes[1].plot(wav_org, sig_org / noi_org, color='green')
    axes[1].axhline(snr_org, ls='--', color='black', label=f'Median SNR={snr_org:.1f}')
    axes[1].set_title(f'SNR per band — organic window {ORGANIC_WINDOW} µm')
    axes[1].set_xlabel('Wavelength (µm)')
    axes[1].set_ylabel('SNR')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('snr_per_band.png', dpi=150)
    print('Saved: snr_per_band.png')

    plt.show()

    # --- Summary for multi-flyby table ---
    print('\n--- COPY THIS INTO YOUR RESULTS TABLE ---')
    print(f'Cube:          {CUBE_PATH}')
    print(f'Altitude:      {ALTITUDE_KM} km')
    print(f'SNR (ref):     {snr_ref:.1f}')
    print(f'SNR (organic): {snr_org:.1f}')