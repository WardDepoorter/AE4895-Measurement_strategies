import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams["font.size"] = 14
import re
import os
import pandas as pd
current_directory = os.getcwd()
intrument_name = "RAMAN"
filetype = "txt"
# File path
file_path_A = current_directory + "/"+ intrument_name+"/SampleA."+filetype
file_path_B = current_directory + "/"+ intrument_name+"/SampleB."+filetype
file_path_C = current_directory + "/"+ intrument_name+"/SampleC."+filetype
file_path_D = current_directory + "/"+ intrument_name+"/SampleD."+filetype
file_path_E = current_directory + "/"+ intrument_name+"/SampleE."+filetype
file_path_F = current_directory + "/"+ intrument_name+"/SampleF."+filetype
file_path_G = current_directory + "/"+ intrument_name+"/SampleG."+filetype
# Read the file
file_path_NaCl_raman = current_directory + "/"+ intrument_name+"/SampleNaCl."+filetype
file_path_NaBr_raman = current_directory + "/"+ intrument_name+"/SampleNaBr."+filetype

def read_spectrum_csv(file_path):
    return pd.read_csv(
        file_path,
        sep=";",          # column separator
        decimal=",",      # decimal separator
        skiprows=1        # skip metadata line
    )
def read_spectrum_txt(file_path):
    return pd.read_csv(
        file_path,
        sep="\t",         # column separator
        decimal=".",      # decimal separator
        skiprows=1        # skip metadata line
    )
if filetype == "csv":
    dataA = read_spectrum_csv(file_path_A)
    dataB = read_spectrum_csv(file_path_B)
    dataC = read_spectrum_csv(file_path_C)
    dataD = read_spectrum_csv(file_path_D)
    dataE = read_spectrum_csv(file_path_E)
    dataF = read_spectrum_csv(file_path_F)
    dataG = read_spectrum_csv(file_path_G)
if filetype == "txt":
    dataA = read_spectrum_txt(file_path_A)
    dataB = read_spectrum_txt(file_path_B)
    dataC = read_spectrum_txt(file_path_C)
    dataD = read_spectrum_txt(file_path_D)
    dataE = read_spectrum_txt(file_path_E)
    dataF = read_spectrum_txt(file_path_F)
    dataG = read_spectrum_txt(file_path_G)
    dataNaCl = read_spectrum_txt(file_path_NaCl_raman)
    dataNaBr = read_spectrum_txt(file_path_NaBr_raman)
# Rename columns for convenience
dataA.columns = ["wavenumber_cm1", "transmittance"]
dataB.columns = ["wavenumber_cm1", "transmittance"]
dataC.columns = ["wavenumber_cm1", "transmittance"]
dataD.columns = ["wavenumber_cm1", "transmittance"]
dataE.columns = ["wavenumber_cm1", "transmittance"]
dataF.columns = ["wavenumber_cm1", "transmittance"]
dataG.columns = ["wavenumber_cm1", "transmittance"]
dataNaCl.columns = ["wavenumber_cm1", "transmittance"]
dataNaBr.columns = ["wavenumber_cm1", "transmittance"]

# Plot
plt.figure()
plt.plot(dataA["wavenumber_cm1"], dataA["transmittance"], label="Sample A")
plt.plot(dataB["wavenumber_cm1"], dataB["transmittance"], label="Sample B")
plt.plot(dataC["wavenumber_cm1"], dataC["transmittance"], label="Sample C")
plt.plot(dataD["wavenumber_cm1"], dataD["transmittance"], label="Sample D")
plt.plot(dataE["wavenumber_cm1"], dataE["transmittance"], label="Sample E")
plt.plot(dataF["wavenumber_cm1"], dataF["transmittance"], label="Sample F")
plt.plot(dataG["wavenumber_cm1"], dataG["transmittance"], label="Sample G")
plt.plot(dataNaCl["wavenumber_cm1"], dataNaCl["transmittance"], label="NaCl")
plt.plot(dataNaBr["wavenumber_cm1"], dataNaBr["transmittance"], label="NaBr")

 
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("counts")
plt.title("Raman Spectra")
plt.gca().invert_xaxis()  # typical for IR spectra

plt.legend()
plt.show()

plt.figure()
plt.plot(dataA["wavenumber_cm1"], dataA["transmittance"], label="Sample A")
plt.plot(dataB["wavenumber_cm1"], dataB["transmittance"], label="Sample B")
# plt.plot(dataC["wavenumber_cm1"], dataC["transmittance"], label="Sample C")
#plt.plot(dataD["wavenumber_cm1"], dataD["transmittance"], label="Sample D")
plt.plot(dataE["wavenumber_cm1"], dataE["transmittance"], label="Sample E")
# plt.plot(dataF["wavenumber_cm1"], dataF["transmittance"], label="Sample F")
# plt.plot(dataG["wavenumber_cm1"], dataG["transmittance"], label="Sample G")
 
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("counts")
plt.title("Raman Spectra")
plt.gca().invert_xaxis()  # typical for IR spectra

plt.legend()
plt.show()