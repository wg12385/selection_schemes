# Size based selection_schemes

import numpy as np

# All schemes return a set of molecule_names from mol_df

# Return the molecule with the most double bonds
def select_molecules_C1(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    atom_df['double_bonds'] = [list(x).count(2) for x in atom_df["conn"]]
    dbl_bonds = atom_df.groupby("molecule_name")["double_bonds"].sum()/2
    sorted = dbl_bonds.sort_values()
    sorted = sorted.index.unique().to_list()
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
# Return the molecule with the lest double bonds
def select_molecules_C2(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    atom_df['double_bonds'] = [list(x).count(2) for x in atom_df["conn"]]
    dbl_bonds = atom_df.groupby("molecule_name")["double_bonds"].sum()/2
    sorted = dbl_bonds.sort_values()
    sorted = sorted.index.unique().to_list()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
def select_molecules_C3(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    atom_df['double_bonds'] = [list(x).count(3) for x in atom_df["conn"]]
    dbl_bonds = atom_df.groupby("molecule_name")["double_bonds"].sum()/3
    sorted = dbl_bonds.sort_values()
    sorted = sorted.index.unique().to_list()
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
# Return the molecule with the lest double bonds
def select_molecules_C4(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    atom_df['double_bonds'] = [list(x).count(3) for x in atom_df["conn"]]
    dbl_bonds = atom_df.groupby("molecule_name")["double_bonds"].sum()/3
    sorted = dbl_bonds.sort_values()
    sorted = sorted.index.unique().to_list()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen