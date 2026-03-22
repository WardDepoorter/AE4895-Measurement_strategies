import matplotlib.pyplot as plt
import os
import pyvims
from pyvims import VIMS
import numpy as np
import re
current_dir = os.getcwd()

#use conda base env, which has pyvims installed, to run this code.

def load_vims_cube(cube, pixel): 
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
    spectra = cube[pixel[1], pixel[0], :]  
    return 


def read_vims_qub(filepath):
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

cube_raw, nl, ns, nb = read_vims_qub('v1234567890_1.qub')