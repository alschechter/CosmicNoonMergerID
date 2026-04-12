import MockImageFunctions_v2
#import astropy.io.fits as pyfits
#import numpy as np
import pandas as pd
import sys
import os
from multiprocessing import Pool
from glob import glob

# badlist = MockImageFunctions.MakeBadList('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mosaics/mosaics/hlsp_candels_hst_wfc3_cos-tot_f125w_v1.0_drz.fits',\
#                                          '/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/CANDELS_RAdecOnly.csv')

# n = int(sys.argv[1])
# m = n-1
z = 1.5

if len(glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground/*.npy')) > 0:
    for f in glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground/*.npy'):
        os.remove(f)


Subhalos_and_Viewpoints = pd.read_csv('SubhaloListForMakeMocks40.csv')  

def makethemocks(m):
    
    #MAKING AN EXAMPLE FIGURE
    #Subhalos_and_Viewpoints = Subhalos_and_Viewpoints#[100:130]
    # for m in range(len(Subhalos_and_Viewpoints)):
    #     print(len(Subhalos_and_Viewpoints))
    #  #MAKING AN EXAMPLE
    # for m in range(5)
    row = Subhalos_and_Viewpoints.iloc[m]
    try:
        MockImageFunctions_v2.CombineSteps(row['Subfind_ID'], row['Viewpoint'], 'wfc3_ir_f125w', z)
        MockImageFunctions_v2.CombineSteps(row['Subfind_ID'], row['Viewpoint'], 'acs_wfc_f606w', z)
        MockImageFunctions_v2.CombineSteps(row['Subfind_ID'], row['Viewpoint'], 'wfc3_ir_f160w', z)
        MockImageFunctions_v2.CombineSteps(row['Subfind_ID'], row['Viewpoint'], 'acs_wfc_f814w', z)

    except Exception as e:
        print(f"Failed to process index {m}: {e}")
    # MockImageFunctions.CombineSteps(Subhalos_and_Viewpoints.iloc[m]['Subfind_ID'], Subhalos_and_Viewpoints.iloc[m]['Viewpoint'], 'wfc3_ir_f125w', z)
    # MockImageFunctions.CombineSteps(Subhalos_and_Viewpoints.iloc[m][1], Subhalos_and_Viewpoints.iloc[m][2], 'acs_wfc_f606w', z)
    # MockImageFunctions.CombineSteps(Subhalos_and_Viewpoints.iloc[m][1], Subhalos_and_Viewpoints.iloc[m][2], 'wfc3_ir_f160w', z)
    # MockImageFunctions.CombineSteps(Subhalos_and_Viewpoints.iloc[m][1], Subhalos_and_Viewpoints.iloc[m][2], 'acs_wfc_f814w', z)

if __name__ == "__main__":
    with Pool(processes=112) as pool:  # choose number of cores
        pool.map(makethemocks, range(len(Subhalos_and_Viewpoints)))





# z = 0.2
# Subhalos_and_Viewpoints = pd.read_csv('SubhaloListForMakeMocks84.csv')
# for m in range(len(Subhalos_and_Viewpoints)):
#     #  #MAKING AN EXAMPLE
#     # for m in range(5)
#     MockImageFunctions.CombineSteps(Subhalos_and_Viewpoints.iloc[m]['Subfind_ID'], Subhalos_and_Viewpoints.iloc[m]['Viewpoint'], 'wfc3_ir_f125w', z)
#     MockImageFunctions.CombineSteps(Subhalos_and_Viewpoints.iloc[m][1], Subhalos_and_Viewpoints.iloc[m][2], 'acs_wfc_f606w', z)
#     MockImageFunctions.CombineSteps(Subhalos_and_Viewpoints.iloc[m][1], Subhalos_and_Viewpoints.iloc[m][2], 'wfc3_ir_f160w', z)
#     MockImageFunctions.CombineSteps(Subhalos_and_Viewpoints.iloc[m][1], Subhalos_and_Viewpoints.iloc[m][2], 'acs_wfc_f814w', z)

