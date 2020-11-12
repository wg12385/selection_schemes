# Size based selection_schemes

import numpy as np


# All schemes return a set of molecule_names from mol_df

# Return the biggest molecules 
def select_molecules_B1(mol_df, atom_df, pair_df, prev_chosen=[], num=3):
    
    
    all_names = mol_df.molecule_name.unique()
    sizes = []
    sizes = atom_df.groupby("molecule_name")["atom_index"].max()
    sorted = sizes.sort_values()
    sorted = sorted.index.unique().to_list()
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
# Return the smallest molecules
def select_molecules_B2(mol_df, atom_df, pair_df, prev_chosen=[], num=3):
    
    
    all_names = mol_df.molecule_name.unique()
    sizes = []
    sizes = atom_df.groupby("molecule_name")["atom_index"].max()
    sorted = sizes.sort_values()
    sorted = sorted.index.unique().to_list()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen