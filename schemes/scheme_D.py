# Random selection_scheme

import numpy as np


# Select molecules with most H atoms
def select_molecules_D1(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    sizes = []
    atom_df['count'] = atom_df['typestr'].mask(atom_df['typestr'].ne("H"))
    sizes = atom_df.groupby('molecule_name')['count'].count()
    sorted = sizes.sort_values()
    sorted = sorted.index.unique().to_list()
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
# Select molecules with most C atoms
def select_molecules_D2(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    sizes = []
    atom_df['count'] = atom_df['typestr'].mask(atom_df['typestr'].ne("C"))
    sizes = atom_df.groupby('molecule_name')['count'].count()
    sorted = sizes.sort_values()
    sorted = sorted.index.unique().to_list()
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
# Select molecules with most N atoms
def select_molecules_D3(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    sizes = []
    atom_df['count'] = atom_df['typestr'].mask(atom_df['typestr'].ne("N"))
    sizes = atom_df.groupby('molecule_name')['count'].count()
    sorted = sizes.sort_values()
    sorted = sorted.index.unique().to_list()
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
# Select molecules with most O atoms
def select_molecules_D4(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    sizes = []
    atom_df['count'] = atom_df['typestr'].mask(atom_df['typestr'].ne("O"))
    sizes = atom_df.groupby('molecule_name')['count'].count()
    sorted = sizes.sort_values()
    sorted = sorted.index.unique().to_list()
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
# Select molecules with most F atoms
def select_molecules_D5(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    sizes = []
    atom_df['count'] = atom_df['typestr'].mask(atom_df['typestr'].ne("F"))
    sizes = atom_df.groupby('molecule_name')['count'].count()
    sorted = sizes.sort_values()
    sorted = sorted.index.unique().to_list()
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen
    
# Select molecules with most non [H, C, N, O, F] atoms
def select_molecules_D6(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    all_names = mol_df.molecule_name.unique()
    new_chosen = []
    sizes = []
    for molname in all_names:
        count = 0
        types = atom_df.loc[(atom_df.molecule_name == molname)]['typestr'].values
        sizes.append(len([x for x in types if x not in ['H', 'C', 'N', 'O', 'F']]))
    sizes = np.asarray(sizes)
    idx = np.argsort(sizes)
    sorted = list(all_names[idx])
    sorted.reverse()
    new_chosen = [x for x in sorted if not x in prev_chosen][:num]
    chosen = prev_chosen

    chosen.extend(new_chosen)
    
    return chosen