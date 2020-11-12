# Random selection_scheme

import numpy as np


# All schemes return a set of molecule_names from mol_df
def select_molecules_A(mol_df, atom_df, pair_df, prev_chosen=[], num=3):
    
    molnames = mol_df.loc[[not x in prev_chosen for x in mol_df.molecule_name]].molecule_name.unique()
    
    chosen = prev_chosen
    new_chosen = np.random.choice(molnames, num, replace=False)
    chosen.extend(new_chosen)
    
    return chosen