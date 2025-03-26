import  pandas as pd
import lc_module as lm
import numpy as np
import gc
import sys
############################### lightcone configuration information begins here ########################################################### ###########################################################################################################################################
BoxLength = 2000.
## compute theta_max given a square_coverage_area
Area_in_square_degrees = 50 ## A is the area coverage in square degrees corresponds to the Euclid Deep Survey

#define the choose direction as a vector
ra = np.radians(31) 
dec  = np.radians(10)

box_info = {"Box1":[0,0,0],"Box2":[BoxLength,0,0],"Box3":[BoxLength,BoxLength,0],"Box4":[2*BoxLength,BoxLength,0]}
### Step 1: Read the table to make the JOIN
df = pd.read_pickle('join_table.pkl')
path_to_files = '/data/astro/scratch/sramakri/uchuu_selected_region/'

############################## lightcone config information ends here #####################################################################
dataframes = []
for index in range(93):
    pid,x,y,z,mvir = lm.generate_lightcone_in_an_interval(index,BoxLength,Area_in_square_degrees,ra,dec,box_info,df,path_to_files)
    dataframe = pd.DataFrame({'pid':pid,'x':x,'y':y,'z':z,'mvir':mvir})
    dataframes.append(dataframe)
result = pd.concat(dataframes, ignore_index=True)
result.to_parquet(path_to_files+'lightcone_catalog.parquet', engine='pyarrow', compression='snappy')
