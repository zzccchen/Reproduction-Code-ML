import os
import json
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.vasp.outputs import Oszicar,Vasprun
import numpy as np
from pylab import *
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
import pymatgen.core as mg
from pymatgen.core.periodic_table import Element
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def target_atom_number(struct):
    for i in range(len(struct)):
        if str(struct[i]).split()[-1] == 'H':
            return i

def prase_dir_ads_energy(adsorption_structure):
    dirs, slab_H_energy_list,Eads = [],[],[]

    nn,all_distance,weight_info = [],[],[]

    distance = []
    weight = {}
    cn = VoronoiNN(tol=0, targets=None, cutoff=13.0, allow_pathological=True,
           weight='solid_angle', extra_nn_info=True)

    struct = adsorption_structure

    target_vornoi = cn.get_nn_info(struct,target_atom_number(struct))
    cnt = 1
    for v in target_vornoi:
        ele = list(v['site'].as_dict()['species'].keys())[0]
        if ele == 'H':
            continue
        distance.append(v['poly_info']['face_dist']*2)
        weight[f'{ele}_{cnt}'] = v['weight']
        cnt += 1
    nn.append(len(distance))
    all_distance.append(distance)
    weight_info.append(weight)

    save_dict = {
        'CN':nn,
        'all_distance':all_distance,
        'weight_info':weight_info
    }
        
    return pd.DataFrame(save_dict)


def generate_features(df):
    df_new = pd.DataFrame()

    # Coordination number
    df_new['CN'] = df['CN']

    # Minimum distance and average distance
    md = []
    ad = []
    for i in df['all_distance']:
        md.append(min(i))
        ad.append(np.mean(i))
    df_new['MD'] = md
    df_new['AD'] = ad

    # Weighted electronegativity
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).X
        # print()
        # res = temp/np.sum(list(weight_dict.values()))
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['EN'] = c

    # 1,2 nearest neighbor electronegativity
    temp1 = []
    temp2 = []
    for i in df['weight_info']:
        i = sorted(i.items(),key=lambda x:x[1],reverse=True)
        ele1 = i[0][0].split('_')[0]
        if len(i) > 1:
            ele2 = i[1][0].split('_')[0]
            temp2.append(Element(ele2).X)
        else:
            temp2.append(0)
        temp1.append(Element(ele1).X)

    df_new['1EN'] = temp1
    df_new['2EN'] = temp2

    # Absolute value of the difference between 1 and 2 near neighbor electronegativity
    c = []
    for i in df['weight_info']:
        i = sorted(i.items(),key=lambda x:x[1],reverse=True)
        if len(i) == 1:
            c.append(0)
        else:
            ele1 = i[0][0].split('_')[0]
            ele2 = i[1][0].split('_')[0]
            wfd = abs(Element(ele1).X-Element(ele2).X)
            c.append(wfd)
    df_new['END'] = c

    # Weight relative to atomic mass
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*(mg.Composition(ele_split).weight)
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['RAM'] = c

    # Weighted work function
    c = []
    import json
    with open('data/wf.json','r')as f:
        wf = json.load(f)
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*wf[ele_split]
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['WF'] = c

    temp1 = []
    temp2 = []
    WFD = []

    for i in df['weight_info']:

        i = sorted(i.items(),key=lambda x:x[1],reverse=True)
        ele1 = i[0][0].split('_')[0]
        temp1.append(wf[ele1])
        if len(i) > 1:
            ele2 = i[1][0].split('_')[0]
            temp2.append(wf[ele2])
            WFD.append(abs(wf[ele1]-wf[ele2]))
        else:
            temp2.append(0)
            WFD.append(0)

    df_new['WFD'] = WFD
    df_new['1WF'] = temp1
    df_new['2WF'] = temp2


    # First ionization energy
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).ionization_energies[0]
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['FIE'] = c

    # Outer atomic orbital energy
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*max(Element(ele_split).atomic_orbitals.values())
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['EOO'] = c

    # Electron affinity
    # https://calculla.com/electron_affinity
    import json
    with open('data/periodic_table_complex.json','r',encoding='utf8')as f:
        pt = json.load(f)
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*pt[ele_split]['Electron affinity']
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['EA'] = c

    # Weighted principal family and period
    g = []
    p = []

    with open('data/periodic_table.json','r',encoding='utf8')as f:
        periodic_table = json.load(f)

    for i in df['weight_info']:
        weight_dict = i
        wp = 0.0
        wg = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            wg += weight_dict[ele]*periodic_table[ele_split]['group']
            wp += weight_dict[ele]*eval(periodic_table[ele_split]['period'])
        res_wg = wg/sum(list(weight_dict.values()))
        res_wp = wp/sum(list(weight_dict.values()))
        g.append(res_wg)
        p.append(res_wp)
    df_new['GN'] = g
    df_new['PN'] = p

    # Weighted atomic number
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).Z
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)

    df_new['AN'] = c

    # resistance
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).electrical_resistivity
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['ER'] = c

    # Atomic radius
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).atomic_radius
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['AR'] = c

    # Thermal conductivity
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).thermal_conductivity
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['TC'] = c

    # Melting point and boiling point
    tmp1 = []
    tmp2 = []
    for i in df['weight_info']:
        weight_dict = i
        temp1,temp2 = 0.0, 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp1 += weight_dict[ele]*Element(ele_split).melting_point
            temp2 += weight_dict[ele]*Element(ele_split).boiling_point
        res1 = temp1/sum(list(weight_dict.values()))
        res2 = temp2/sum(list(weight_dict.values()))
        tmp1.append(res1)
        tmp2.append(res2)
    df_new['MP'] = tmp1
    df_new['BP'] = tmp2

    # Bulk modulus
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            try:
                temp += weight_dict[ele]*Element(ele_split).bulk_modulus
            except:
                temp += 0
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['BM'] = c

    # Moore volume
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*Element(ele_split).molar_volume
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['MV'] = c

    # Oxidation state 
    import json
    with open('data/periodic_table.json','r',encoding='utf8')as f:
        pt = json.load(f)
    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            try:
                temp += weight_dict[ele]*pt[ele_split]['ICSD oxidation states'][-1]
            except:
                temp += 0 # For the elements that don't have an oxidation state, let's do it at zero
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['OS'] = c

    # Unoccupied electron number (UOE)
    import json
    with open('data/uoe.json','r',encoding='utf8')as f:
        pt = json.load(f)

    c = []
    for i in df['weight_info']:
        weight_dict = i
        temp = 0.0
        for ele in weight_dict.keys():
            ele_split = ele.split('_')[0]
            temp += weight_dict[ele]*pt[ele_split]
        res = temp/np.sum(list(weight_dict.values()))
        c.append(res)
    df_new['UOE'] = c
    
    return df_new
    


def predict(file_name):
    adsorption_structure = Structure.from_file(file_name)
    df = prase_dir_ads_energy(adsorption_structure)
    df_new = generate_features(df)
    df = pd.read_excel('data/trainset_loop6.xlsx')
    df = df.drop(['PN', 'AN'],axis = 1) 
    Y, X = np.array(df['Eads']), df.drop(['Eads'],axis = 1)
    features = list(X.columns)
    
    scaler = StandardScaler()
    scaler.fit(X) 
    X = scaler.transform(X) 
    
    features = ['CN', 'MD', 'AD', 'EN', '1EN','2EN','END', 'RAM', 'WF', 'WFD',
     '1WF', '2WF', 'FIE', 'EOO', 'EA', 'GN', 'ER', 'AR', 'TC', 'MP', 'BP', 'BM', 'MV', 'OS', 'UOE']

    X_newspace = scaler.transform(df_new[features])

    best_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=0.8, enable_categorical=False,
                 eval_metric='mae', gamma=0, gpu_id=-1, importance_type=None,
                 interaction_constraints='', learning_rate=0.05, max_delta_step=0,
                 max_depth=5, min_child_weight=1, missing=nan,
                 monotone_constraints='()', n_estimators=1000, n_jobs=4,
                 num_parallel_tree=1, predictor='auto', random_state=0,
                 reg_alpha=0.1, reg_lambda=2, scale_pos_weight=1, seed=0,
                 subsample=0.8, tree_method='exact', validate_parameters=1,
                 verbosity=None)

    best_model.fit(X,Y)
    Y_newspace = best_model.predict(X_newspace)
    print(f'The predicted H adsorption energy for the input adsorption strucutre is: {Y_newspace[0]:.2f} eV')
