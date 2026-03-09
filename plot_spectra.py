import pandas as pd
import matplotlib.pyplot as plt
import os
current_directory = os.getcwd()
    
# File path
file_path = current_directory + "/LAB/Administrator 05.csv"

# Read the file
data = pd.read_csv(
    file_path,
    sep=";",          # column separator
    decimal=",",      # decimal separator
    skiprows=1        # skip metadata line
)

# Rename columns for convenience
data.columns = ["wavenumber_cm1", "transmittance"]

# Plot
plt.figure()
plt.plot(data["wavenumber_cm1"], data["transmittance"])

plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance (%)")
plt.title("Spectrum")
plt.gca().invert_xaxis()  # typical for IR spectra

plt.show()