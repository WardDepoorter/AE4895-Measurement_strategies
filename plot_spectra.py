import pandas as pd
import matplotlib.pyplot as plt
import os
current_directory = os.getcwd()
    
# File path
file_path_A = current_directory + "/LAB/Sample A.csv"
file_path_B = current_directory + "/LAB/Sample B.csv"
file_path_C = current_directory + "/LAB/Sample C.csv"
file_path_D = current_directory + "/LAB/Sample D.csv"
file_path_E = current_directory + "/LAB/Sample E.csv"
file_path_F = current_directory + "/LAB/Sample F.csv"
file_path_G = current_directory + "/LAB/Sample G.csv"
# Read the file
def read_spectrum(file_path):
    return pd.read_csv(
        file_path,
        sep=";",          # column separator
        decimal=",",      # decimal separator
        skiprows=1        # skip metadata line
    )
dataA = read_spectrum(file_path_A)
dataB = read_spectrum(file_path_B)
dataC = read_spectrum(file_path_C)
dataD = read_spectrum(file_path_D)
dataE = read_spectrum(file_path_E)
dataF = read_spectrum(file_path_F)
dataG = read_spectrum(file_path_G)

# Rename columns for convenience
dataA.columns = ["wavenumber_cm1", "transmittance"]
dataB.columns = ["wavenumber_cm1", "transmittance"]
dataC.columns = ["wavenumber_cm1", "transmittance"]
dataD.columns = ["wavenumber_cm1", "transmittance"]
dataE.columns = ["wavenumber_cm1", "transmittance"]
dataF.columns = ["wavenumber_cm1", "transmittance"]
dataG.columns = ["wavenumber_cm1", "transmittance"]

# Plot
plt.figure()
plt.plot(dataA["wavenumber_cm1"], dataA["transmittance"], label="Sample A")
plt.plot(dataB["wavenumber_cm1"], dataB["transmittance"], label="Sample B")
plt.plot(dataC["wavenumber_cm1"], dataC["transmittance"], label="Sample C")
plt.plot(dataD["wavenumber_cm1"], dataD["transmittance"], label="Sample D")
plt.plot(dataE["wavenumber_cm1"], dataE["transmittance"], label="Sample E")
plt.plot(dataF["wavenumber_cm1"], dataF["transmittance"], label="Sample F")
plt.plot(dataG["wavenumber_cm1"], dataG["transmittance"], label="Sample G")

plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance (%)")
plt.title("Spectrum")
plt.gca().invert_xaxis()  # typical for IR spectra

plt.legend()
plt.show()