import matplotlib.pyplot as plt
import os
import pyvims
from pyvims import VIMS
import numpy as np
import re
current_dir = os.getcwd()

#use conda base env, which has pyvims installed, to run this code.

def read_calibrated_cube(cube, pixel): 
    """
    Load VIMS spectra for a given pixel from a VIMS cube.

    Parameters:
    cube (str): Path to the VIMS cube file.
    pixel (tuple): A tuple containing the (x, y) coordinates of the pixel.

    Returns:
    np.ndarray: The spectra for the specified pixel.
    """
    # Load the VIMS cube
    cube = VIMS(cube)
    # Extract the spectra for the specified pixel
    spectrum = cube[pixel[0], pixel[1]].spectrum
    wavelengths = cube.wvlns
    plt.figure(figsize=(12, 10))

    plt.imshow(cube@2.03, extent=cube.extent, cmap='gray', vmin=0, vmax=.25)

    plt.colorbar(extend='max', label='I/F')

    #you can place few points on the object to choose which VIMS spectra you would like to visualise
    plt.scatter( pixel[0],pixel[1], s=150)

    plt.xlabel(cube.slabel)
    plt.ylabel(cube.llabel)

    plt.xticks(cube.sticks)
    plt.yticks(cube.lticks)
    plt.show()   
    return wavelengths, spectrum


def read_raw_cube(filepath):
    """
    Read a raw Cassini VIMS .qub file directly.
    Returns the data cube as a numpy array and the wavelength axis.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    # The label is ASCII at the start of the file
    # Find where it ends (PDS labels end with "END\r\n")
    label_end = raw.find(b'END\r\n') + 5
    label_text = raw[:label_end].decode('ascii', errors='ignore')
    
    # Parse key parameters from label
    def get_label_val(key):
        match = re.search(rf'{key}\s*=\s*(\S+)', label_text)
        return match.group(1).strip('()') if match else None
    
    n_lines   = int(get_label_val('LINES'))
    n_samples = int(get_label_val('LINE_SAMPLES'))
    n_bands   = int(get_label_val('BANDS'))         # usually 352
    record_bytes = int(get_label_val('RECORD_BYTES'))
    label_records = int(get_label_val('LABEL_RECORDS'))
    
    # Data starts after the label records
    data_offset = label_records * record_bytes
    
    # Read binary data - raw DN values are 16-bit integers
    # Axis order in file: (Line, Band, Sample)
    n_total = n_lines * n_bands * n_samples
    data_raw = np.frombuffer(raw[data_offset:data_offset + n_total*2],
                              dtype='>i2')  # big-endian 16-bit int
    
    # Reshape to (Lines, Bands, Samples) then reorder to (Lines, Samples, Bands)
    cube = data_raw.reshape((n_lines, n_bands, n_samples))
    cube = np.transpose(cube, (0, 2, 1))  # -> (Lines, Samples, Bands)
    
    return cube, n_lines, n_samples, n_bands
cube_name = '1487298702_1'
filename = current_dir + '/raw_flyby_data/v' + cube_name + '.qub'
print(f"Reading raw cube from: {filename}")
# cube, n_lines, n_samples, n_bands = read_raw_cube(filename)
# print(f"Cube shape: {cube.shape} (Lines: {n_lines}, Samples: {n_samples}, Bands: {n_bands})")
wavelengths, calibrated_spectrum = read_calibrated_cube(cube_name, (24, 32))

# Plot the calibrated spectrum
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, calibrated_spectrum, label='Calibrated Spectrum', color='blue')
plt.xlabel('Wavelength (µm)')
plt.ylabel('I/F')
plt.title('Calibrated VIMS Spectrum at Pixel (20, 20)')
plt.legend()
plt.grid()
plt.show()
