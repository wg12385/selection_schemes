
from model.gtn_model import GTNmodel
import model.features.graph_input as graph_in
import glob
import numpy as np
import pandas as pd

import pickle

import tqdm
import time

import os
import sys

def add_IMP_to_df(atom_df, pair_df, model_dir):

    mol_df, graphs = graph_in.make_graph_df(atom_df, pair_df)
    
    print('Making predictions:')

    modelfile = model_dir + 'all_model.torch'
    
    model = GTNmodel()
    model.load_model(modelfile)
    model.params['batch_size'] = 4
    test_loader = model.get_input(graphs, mol_df)
    graphs_out = model.predict(test_loader, progress=True)

    atom_df, pair_df = model.assign_preds(graphs_out, mol_df, atom_df, pair_df, assign_to="", progress=True)

    return atom_df, pair_df

if __name__ == "__main__":
    
    pickle_dir = sys.argv[1]
    model_dir = sys.argv[2]
    
    atom_file = pickle_dir + 'atoms.pkl'
    pair_file = pickle_dir + 'pairs.pkl'
    
    assert os.path.isfile(atom_file), print(atom_file)
    assert os.path.isfile(pair_file), print(atom_file)
    
    atom_df = pd.read_pickle(atom_file)
    pair_df = pd.read_pickle(pair_file)
    
    atom_df, pair_df = add_IMP_to_df(atom_df, pair_df, model_dir)
    
    pickle.dump(atom_df, open(atom_file, 'wb'))
    pickle.dump(pair_df, open(pair_file, 'wb'))