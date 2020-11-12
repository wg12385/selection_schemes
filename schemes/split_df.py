import copy

def get_split(chosen, mol_df, graphs):
    
    copy_mol_df = copy.copy(mol_df)
    train_mol_df = copy_mol_df.loc[[x in chosen for x in copy_mol_df.molecule_name]]
    test_mol_df = copy_mol_df.drop(train_mol_df.index)
    
    train_graphs = []
    test_graphs = []
    for molname in train_mol_df.molecule_name:
        assert molname not in list(test_mol_df.molecule_name.unique())   
        idx = train_mol_df.loc[(train_mol_df.molecule_name == molname)]['index'].to_numpy()[0]
        train_graphs.append(graphs[idx])
    for molname in test_mol_df.molecule_name:
        assert molname not in list(train_mol_df.molecule_name.unique())
        idx = test_mol_df.loc[(test_mol_df.molecule_name == molname)]['index'].to_numpy()[0]
        test_graphs.append(graphs[idx])
    
    return train_graphs, train_mol_df, test_graphs, test_mol_df