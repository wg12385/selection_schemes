# Copyright 2020 Will Gerrard, Calvin Yiu
#This file is part of autoenrich.

#autoenrich is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#autoenrich is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with autoenrich.  If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import shutil
sys.path.append(os.path.realpath(os.path.dirname(__file__))+'/../')
import pandas as pd
import numpy as np
import itertools

import model.features.graph_input as graphin

from schemes.scheme_E import *

from schemes.split_df import get_split

import glob
import copy


def cleanup():
    files_to_delete = glob.glob('*.txt')
    files_to_delete.extend(glob.glob('*.torch'))
    for file in files_to_delete:
        os.remove(file)
    shutil.rmtree("runs")

# E1 is FPS on ecfp4
def test_scheme_E1():

    # Need to test it as a learning curve, not just pick one

    num = 3

    atom_df = pd.read_pickle('tests/test_mols/atoms.pkl')
    pair_df = pd.read_pickle('tests/test_mols/pairs.pkl')
    mol_df, graphs = graphin.make_graph_df(atom_df, pair_df)

    total = len(graphs)
    counter = 0
    chosen = []
    while len(chosen) < len(graphs) - num:
        
        # Iterations continue until there are not enough graphs to select
        # Each iteration we want (num*n) graphs in train, and (total-(num*n)) graphs in test
        # Make the selection the same each time, we just want to select chosen from the df
        # We then just grow chosen each time using the selection scheme
        
        counter += 1
        chosen = select_molecules_E1(mol_df, atom_df, pair_df, prev_chosen=chosen, num=num)
        train_graphs, train_mol_df, test_graphs, test_mol_df = get_split(chosen, mol_df, graphs)


        assert len(train_graphs) == num * counter
        assert len(test_graphs) == total - (num * counter)
        assert len(train_mol_df.molecule_name.unique()) == len(train_mol_df.molecule_name)
        assert len(test_mol_df.molecule_name.unique()) == len(test_mol_df.molecule_name)
        assert counter < total/num
        
        for molname in train_mol_df.molecule_name.unique():
            assert not molname in test_mol_df.molecule_name.unique()
    assert counter == 2
    assert len(chosen) == len(graphs) - len(graphs)%3