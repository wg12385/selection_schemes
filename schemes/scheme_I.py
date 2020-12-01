# Size based selection_schemes

import numpy as np

from rdkit import DataStructs

# Get FEP values for all molecules not already picked
def get_FEPs(atom_df, prev_chosen):
    molnames = []
    feps = []
    for molname in atom_df.molecule_name.unique():
        if molname in molnames or molname in prev_chosen:
            continue
        fep = atom_df.loc[atom_df.molecule_name == molname]['predicted_ic50'].values[0]
        feps.append(fep)
        molnames.append(molname)


    return molnames, np.array(feps)

# Get FEP values for all molecules not already picked
def get_FEP_varss(atom_df, prev_chosen):
    molnames = []
    feps = []
    for molname in atom_df.molecule_name.unique():
        if molname in molnames or molname in prev_chosen:
            continue
        fep = atom_df.loc[atom_df.molecule_name == molname]['predicted_ic50_var'].values[0]
        feps.append(fep)
        molnames.append(molname)


    return molnames, np.array(feps)
        
        
    
# Return the molecules with the lowest predicted FEP
def select_molecules_I1(mol_df, atom_df, pair_df, prev_chosen=[], num=3):
    
    # Assumption here is that the atom_df will have the FEP values added
    # Will have to write some other script to do this, shouldn't be too hard
    # FEP values stored in atom_df['predicted_ic50']
    molnames, feps = get_FEPs(atom_df, prev_chosen)
    
    # Sort FEPS from lowest to highest
    idx = np.argsort(feps)
    chosen_mols = [molnames[x] for x in idx[:num]]
    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    
# Return the molecules with the highest predicted FEP
def select_molecules_I2(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    molnames, feps = get_FEPs(atom_df, prev_chosen)
    
    # Sort FEPS from lowest to highest
    idx = np.argsort(feps)
    # reverse
    np.flip(idx)
    chosen_mols = [molnames[x] for x in idx[:num]]
    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    
# Return the molecules covering the spread of FEP predictions
def select_molecules_I3(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    molnames, feps = get_FEPs(atom_df, prev_chosen)
    
    ideal_values = np.linspace(np.min(feps), np.max(feps), num)
    
    idx = []
    for value in ideal_values:
        choose = -1
        closest = -1
        for i, fep in enumerate(feps):
            if np.absolute(value-fep) < np.absolute(value-closest):
                choose = i
                closest = fep
        idx.append(choose)
    
    chosen_mols = [molnames[x] for x in idx]
    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    
# Return the molecules matching the FEP prediction distribution
def select_molecules_I4(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    molnames, feps = get_FEPs(atom_df, prev_chosen)

    distances = []
    for f, fep in enumerate(feps):
        distances.append(np.sum(np.absolute(fep - feps)))
        
    # Use the distances as inverse probability:
    # points with more points close to them are more likely to be picked
    distances = (np.max(distances) - distances)
    distances = distances / (np.sum(distances))

    idx = np.random.choice(np.arange(0, len(distances)), num, p=distances, replace=False)
    chosen_mols = [molnames[x] for x in idx[:num]]
    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    
# Return the molecules matching the inverse of the FEP prediction distribution
# Effectively the same as FPS
def select_molecules_I5(mol_df, atom_df, pair_df, prev_chosen=[], num=3):

    molnames, feps = get_FEPs(atom_df, prev_chosen)

    distances = []
    for f, fep in enumerate(feps):
        distances.append(np.sum(np.absolute(fep - feps)))
        
    # Use the distances as probability:
    # points with less points close to them are more likely to be picked
    distances = distances / (np.sum(distances))

    idx = np.random.choice(np.arange(0, len(distances)), num, p=distances, replace=False)
    chosen_mols = [molnames[x] for x in idx[:num]]
    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    
# Return the molecules with the lowest predicted FEP variance
def select_molecules_I6(mol_df, atom_df, pair_df, prev_chosen=[], num=3):
    
    molnames, feps = get_FEP_vars(atom_df, prev_chosen)
    
    # Sort FEPS from lowest to highest
    idx = np.argsort(feps)
    chosen_mols = [molnames[x] for x in idx[:num]]
    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    
# Return the molecules with the lowest predicted FEP variance
def select_molecules_I7(mol_df, atom_df, pair_df, prev_chosen=[], num=3):
    
    molnames, feps = get_FEP_vars(atom_df, prev_chosen)
    
    # Sort FEPS from lowest to highest
    idx = np.argsort(feps)
    np.flip(idx)
    chosen_mols = [molnames[x] for x in idx[:num]]
    chosen_mols.extend(prev_chosen)
    
    return chosen_mols
    