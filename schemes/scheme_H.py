# Size based selection_schemes

import numpy as np

from rdkit import DataStructs

def get_targets():
    targets = ['CCS', 'HCS', '1JCH', '1JCC', '2JCH', '2JCC',
            '2JHH', '3JCH', '3JCC', '3JHH']
    return targets


def make_imp_fingerprint(atom_df, pair_df, mol_df):
    
    targets = get_targets()
    
    molecule_names = mol_df.molecule_name.unique()
    
    fp_dict = {}
    
    for name in molecule_names:
        fingerprint = {}
        for target in targets:
            if len(target) == 3:
                values = atom_df.loc[(atom_df.molecule_name == name)
                                & (atom_df.typestr == target[0])]['shift'].to_numpy()
            elif len(target) == 3:
                values = pair_df.loc[(pair_df.molecule_name == name)
                                & (pair_df.nmr_types == target)]['coupling'].to_numpy()
            if len(values) == 0:
                continue
            fingerprint[target] = values
        fp_dict[name] = fingerprint
        
    return fp_dict
    
    

def FingerprintSimilarity(fp1, fp2):
    
    sims = []
    
    for target in fp1.keys():
        if target not in fp2.keys():
            continue
        
        if len(fp1[target]) == len(fp2[target]):
            sim = np.linalg.norm(fp1[target]-fp2[target])
        else:
            if len(fp1[target]) < len(fp2[target]):
                matched = []
                for value in fp1[target]:
                    idx = (np.abs(fp2[target] - value)).argmin()
                    matched.append(fp2[target][idx])
                sim = np.linalg.norm(matched-fp1[target])
            else:
                matched = []
                for value in fp2[target]:
                    idx = (np.abs(fp1[target] - value)).argmin()
                    matched.append(fp1[target][idx])
                sim = np.linalg.norm(matched-fp2[target])
        sims.append(np.log(sim)+10)
    
    return np.sum(sims)

# Return the molecules with the least similar IMP prediction fingerprints
def select_molecules_H1(mol_df, atom_df, pair_df, prev_chosen=[], num=3):
    
    fp_dict = make_imp_fingerprint(atom_df, pair_df, mol_df)
        
    random_candidates = [x for x in fp_dict.keys() if x not in prev_chosen]
    all_molnames = mol_df.molecule_name.unique()
    
    init_mol = np.random.choice(random_candidates)
    chosen_mols = [init_mol]
    
    while len(chosen_mols) < num:
        biggest_dist = 0.0
        for molid1 in all_molnames:
            if molid1 in chosen_mols or molid1 in prev_chosen:
                continue
            dist = 0.0
            for molid2 in all_molnames:
                if molid2 not in chosen_mols or molid2 in prev_chosen:
                    continue
  
                dist += FingerprintSimilarity(fp_dict[molid1], fp_dict[molid2])
            
            print(molid1, molid2, dist, biggest_dist)
            
            if dist > biggest_dist:
                biggest_dist = dist
                candidate = molid1
                
        chosen_mols.append(candidate)

    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    