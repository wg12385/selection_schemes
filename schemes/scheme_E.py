# Size based selection_schemes

import numpy as np
import mol_translator.imp_converter.dataframe_read as df_read

from rdkit import DataStructs

# All schemes return a set of molecule_names from mol_df

# Return the molecules with the least similar ecfp4 fingerprints
def select_molecules_E1(mol_df, atom_df, pair_df, prev_chosen=[], num=3):
    
    mols = df_read.read_df(atom_df, pair_df)
    random_candidates = []
    for mol in mols:
        if mol.info['molid'] in prev_chosen:
            continue
        
        try:
            mol.get_rdkit_fingerprint()
        except:
            mol.mol_properties['ecfp4'] = None
            
        random_candidates.append(mol.info['molid'])
        
    
    init_mol = np.random.choice(random_candidates)
    chosen_mols = [init_mol]
    
    while len(chosen_mols) < num:
        biggest_dist = 0.0
        for m1, mol1 in enumerate(mols):
            if mol1.info['molid'] in chosen_mols or mol1.info['molid'] in prev_chosen:
                continue
            dist = 0.0
            for m2, mol2 in enumerate(mols):
                if mol2.info['molid'] not in chosen_mols or mol2.info['molid'] in prev_chosen:
                    print('skip because')
                    continue

                if mol1.mol_properties['ecfp4'] == None or mol2.mol_properties['ecfp4'] == None:
                    dist += 0.000001
                else:    
                    dist += DataStructs.FingerprintSimilarity(mol1.mol_properties['ecfp4'], mol2.mol_properties['ecfp4'])
            if dist > biggest_dist:
                biggest_dist = dist
                candidate = mol1.info['molid']
                
        chosen_mols.append(candidate)

    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    