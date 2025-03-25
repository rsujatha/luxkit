from pathlib import Path
import numpy as np
import pandas as pd
from line_profiler import profile

@profile
def radial_dist(data):
    return np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)

@profile
def halo_reader(path_to_files,Total_no_of_Boxes,box_info,jointable,halodir_string):
    meta = jointable

    haloL = []
    for i in range(len(box_info)):
        if not meta[f'Box{i+1}(w/ buffer)']:
            continue

        file_path = Path(path_to_files) / f'halodir_{jointable[halodir_string]}_box{i+1}.parquet'
        print (file_path)
        data = pd.read_parquet(file_path, columns=['id','pid','upid','scale_of_last_mm','x', 'y', 'z','xoff','vx','vy','vz','c_to_a','b_to_a','vrms','vmax','vpeak','a_x','a_y','a_z','jx','jy','jz','spin','rs','rvir','rvmax','acc_rate_inst','acc_rate_1_tdyn','num_prog', 'last_mainleaf_depthfirst_id','m200b','mvir','m200c','desc_id'])
        data['x'] += box_info[f'Box{i+1}'][0]
        data['y'] += box_info[f'Box{i+1}'][1]
        data['z'] += box_info[f'Box{i+1}'][2]
        data['r_dist'] = radial_dist(data)

        data['inner_prod'] = (data['x']*meta['direction_vector_x'] + data['y']*meta['direction_vector_y'] + data['z']*meta['direction_vector_z']) / \
                             data.r_dist

        inner_prod_lim = np.cos(meta['theta(radians)'] + meta['buffer_theta(radians)'])	
        sub = data[(data.r_dist > meta['comovD_Min(Mpchinv)']-1) & \
                   (data.r_dist < meta['comovD_Max(Mpchinv)'] +1)& \
                   (data.inner_prod > inner_prod_lim)]

        haloL.append(sub)

    res = pd.concat(haloL, ignore_index=True)

    return res

@profile
def interpolate(chistar,chi_f,chi_i,q_f,q_i,interpolation_type):
    if interpolation_type=='lin':
        return q_i + (chistar-chi_i)/(chi_f-chi_i)*(q_f-q_i)
    elif interpolation_type=='log':
        return 10**(np.log10(q_i) + (chistar-chi_i)/(chi_f-chi_i)*(np.log10(q_f)-np.log10(q_i)))

@profile
def interpolation_scheme(joined_df,jointable):
    chi_f = jointable['comovD_Min(Mpchinv)']
    chi_i = jointable['comovD_Max(Mpchinv)']
    r_f = np.sqrt(joined_df["x_zf"]**2+joined_df["y_zf"]**2+joined_df["z_zf"]**2)  
    r_i = np.sqrt(joined_df["x_zi"]**2+joined_df["y_zi"]**2+joined_df["z_zi"]**2) 
    V = (r_f - r_i)/(chi_f-chi_i) ## Definining velocity######
    chistar = (r_f - V *chi_f)/(1-V) ## Defining chistar		
    interpolated = {}
    if jointable.field1 == 'last_mainleaf_depthfirst_id':
        interpolated['id'] = joined_df['id_zf']
        interpolated['pid'] = joined_df['pid_zf']
        interpolated['upid'] = joined_df['upid_zf']
        interpolated['scale_of_last_mm'] = joined_df['scale_of_last_mm_zf']
    elif jointable.field1 == 'desc_id':
        interpolated['id'] = joined_df['id_zi']
        interpolated['pid'] = joined_df['pid_zi']
        interpolated['upid'] = joined_df['upid_zi']
        interpolated['scale_of_last_mm'] = joined_df['scale_of_last_mm_zi']
    interpolation_type = {'x':'lin','y':'lin','z':'lin','xoff':'lin','vx':'lin','vy':'lin','vz':'lin','c_to_a':'lin','b_to_a':'lin','vrms':'log','vmax':'log','vpeak':'log','a_x':'log','a_y':'log','a_z':'log','jx':'log','jy':'log','jz':'log','spin':'log','rs':'log','rvir':'log','rvmax':'log','m200b':'log','mvir':'log','m200c':'log','acc_rate_inst':'log','acc_rate_1_tdyn':'log'}
    print (interpolation_type)
    for axis in interpolation_type.keys():
        interpolated[axis] = interpolate(chistar, chi_f, chi_i, joined_df[f"{axis}_zf"], joined_df[f"{axis}_zi"],interpolation_type[axis])
    ### The following "if" condition accounts for the haloes that are born in the final redshift that don't need interpolation
    if jointable['field1'] == 'last_mainleaf_depthfirst_id':
        select = (joined_df["num_prog_zf"]==0)
        for axis in interpolation_type.keys():
            interpolated[axis][select] = joined_df[f"{axis}_zf"]   
####### if the halo in the final snapshot has more than 1 progenitors then more weightage is given to the halo(prog or descendent) being interpolated.
    select = joined_df["num_prog_zf"]>1
    for axis in list(interpolation_type.keys())[3:]:
        if jointable['field1']=='last_mainleaf_depthfirst_id':  
            interpolated[axis][select] = joined_df[f"{axis}_zf"][select]
        elif jointable['field1']=='desc_id':
            interpolated[axis][select] = joined_df[f"{axis}_zi"][select]   
    return interpolated

@profile
def gen_slice(index, BoxLength, Area_in_square_degrees, ra, dec,\
              box_info, df, path_to_files):
    """ Generate the lightcone."""
    
    meta = df
    xmeta = meta.iloc[index]
    cos_theta_max = 1 - (Area_in_square_degrees / 2)*np.pi / 180**2 

    Total_no_of_Boxes = False # No longer used...
    df0 = halo_reader(path_to_files, Total_no_of_Boxes, box_info, xmeta, 'halodir1')
    df1 = halo_reader(path_to_files, Total_no_of_Boxes, box_info, xmeta, 'halodir2')

    dir_vec = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

    meta = df
    xmeta = df.iloc[index]
    test_order = xmeta.halodir1 > xmeta.halodir2

    if test_order:
        newborn = df0[df0.num_prog == 0]
        D1 = meta.iloc[index]['comovD_Min(Mpchinv)']
        D2 = meta.iloc[index+1]['comovD_Max(Mpchinv)'] 
    else:
        newborn = df1[df1["num_prog"]==0]  ## the final snapshot
        D1 = df.iloc[index-1]['comovD_Min(Mpchinv)']
        D2 = df.iloc[index]['comovD_Max(Mpchinv)']

    r_nb = np.sqrt(newborn['x']**2 + newborn['y']**2 + newborn['z']**2)
    xvec = np.array(dir_vec)
    cosangle = (newborn[['x', 'y', 'z']] @ xvec) / r_nb

    # Select based on the geometry.
    sel_geo = (cosangle > cos_theta_max) & (r_nb <= xmeta['comovD_Max(Mpchinv)']) & (r_nb > xmeta['comovD_Min(Mpchinv)'])
    newborn = newborn[sel_geo]
    r_nb = r_nb[sel_geo]

    # Assignment of halo to redshift.
    prob = np.round((D2 - r_nb) / (D2 - D1), 2)
    assert not np.isnan(prob).any(), 'Found NaN values!'

    print('newborn haloes: min probability:',prob.min(),\
          'maximum probability:', prob.max())
    sel_redshift = np.random.binomial(1, prob).astype(bool)
    newborn = newborn[sel_redshift]
    r_nb = r_nb[sel_redshift]

    # Merging the catalogs.
    suffixes = ('_zf', '_zi') if test_order else ('_zi', '_zf')
    comb = df0.merge(df1, how='left', left_on=xmeta['field1'], right_on=xmeta['field2'], suffixes=suffixes)

    print('Before cleaning:', len(comb))
    sel_less = (comb.r_dist_zi < xmeta['comovD_Min(Mpchinv)']) & (comb.r_dist_zf < xmeta['comovD_Min(Mpchinv)'])
    sel_more = (comb.r_dist_zi >= xmeta['comovD_Max(Mpchinv)']) & (comb.r_dist_zf >= xmeta['comovD_Max(Mpchinv)'])

    comb = comb[~(sel_less | sel_more)]
    print('After cleaning:', len(comb))

    # Linear interpolation of positions.
    interp = interpolation_scheme(comb, xmeta)

    # Lightcone selection.
    r = np.sqrt(interp['x']**2 + interp['y']**2 + interp['z']**2)
    cosangle = (np.array([interp['x'],interp['y'],interp['z']]).T @ xvec) / r
    sel_lc = (cosangle > cos_theta_max) & (r <= xmeta['comovD_Max(Mpchinv)']) & (r > xmeta['comovD_Min(Mpchinv)']) \
             & (comb['num_prog_zf'] != 0)

    res = pd.DataFrame(interp)[sel_lc]
#    lightcone = comb[sel_lc]

#    if xmeta.field1 == 'last_mainleaf_depthfirst_id':
#        pid = lightcone['pid_zf']
#    elif xmeta.field1 == 'desc_id':
#        pid = lightcone['pid_zi']

 #   res = pd.DataFrame({'pid': pid.values, 'x': pos_x_interp1[sel_lc].values, 'y': pos_y_interp1[sel_lc].values,
#                        'z': pos_z_interp1[sel_lc].values, 'vx':vel_x_ip[sel_lc].values, 'mvir': mvir_interp1[sel_lc].values})
    res = pd.concat([res, newborn[['id','pid','upid','scale_of_last_mm','x', 'y', 'z','xoff','vx','vy','vz','c_to_a','b_to_a','vrms','vmax','vpeak','a_x','a_y','a_z','jx','jy','jz','spin','rs','rvir','rvmax','m200b','mvir','m200c','acc_rate_inst','acc_rate_1_tdyn']]], ignore_index=True)
    return res