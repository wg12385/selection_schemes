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

from schemes.scheme_D import *

from schemes.split_df import get_split

import glob
import copy

'''
    THERES NO HYDROGENS IN THE STRUCTURES !!!!!!!!!

'''


def cleanup():
    files_to_delete = glob.glob('*.txt')
    files_to_delete.extend(glob.glob('*.torch'))
    for file in files_to_delete:
        os.remove(file)
    shutil.rmtree("runs")
    

# Pick max H to lowest H
def test_scheme_D1():
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
    
        chosen = select_molecules_D1(mol_df, atom_df, pair_df, prev_chosen=chosen, num=num)
        train_graphs, train_mol_df, test_graphs, test_mol_df = get_split(chosen, mol_df, graphs)

        assert len(train_graphs) == num * counter
        assert len(test_graphs) == total - (num * counter)
        assert len(train_mol_df.molecule_name.unique()) == len(train_mol_df.molecule_name)
        assert len(test_mol_df.molecule_name.unique()) == len(test_mol_df.molecule_name)
        assert counter < total/num
        
        for molname1 in train_mol_df.molecule_name:
            mol_df1 = atom_df.loc[(atom_df.molecule_name == molname1)]["typestr"]

            for molname2 in test_mol_df.molecule_name:
                mol_df2 = atom_df.loc[(atom_df.molecule_name == molname1)]["typestr"]
                count1 = list(mol_df1.values).count("H")
                count2 = list(mol_df2.values).count("H")

                assert count1 >= count2

                
    
# Pick max C to lowest C
def test_scheme_D2():
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
    
        chosen = select_molecules_D2(mol_df, atom_df, pair_df, prev_chosen=chosen, num=num)
        train_graphs, train_mol_df, test_graphs, test_mol_df = get_split(chosen, mol_df, graphs)
        print(atom_df)
        assert len(train_graphs) == num * counter
        assert len(test_graphs) == total - (num * counter)
        assert len(train_mol_df.molecule_name.unique()) == len(train_mol_df.molecule_name)
        assert len(test_mol_df.molecule_name.unique()) == len(test_mol_df.molecule_name)
        assert counter < total/num
        
        for molname1 in train_mol_df.molecule_name:
            mol_df1 = atom_df.loc[(atom_df.molecule_name == molname1)]["typestr"]
            count1 = list(mol_df1.values).count("C")
            for molname2 in test_mol_df.molecule_name:
                mol_df2 = atom_df.loc[(atom_df.molecule_name == molname2)]["typestr"]
                count1 = list(mol_df1.values).count("C")
                count2 = list(mol_df2.values).count("C")
                assert count1 >= count2
# Pick max N to lowest N
def test_scheme_D3():
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
    
        chosen = select_molecules_D3(mol_df, atom_df, pair_df, prev_chosen=chosen, num=num)
        train_graphs, train_mol_df, test_graphs, test_mol_df = get_split(chosen, mol_df, graphs)

        assert len(train_graphs) == num * counter
        assert len(test_graphs) == total - (num * counter)
        assert len(train_mol_df.molecule_name.unique()) == len(train_mol_df.molecule_name)
        assert len(test_mol_df.molecule_name.unique()) == len(test_mol_df.molecule_name)
        assert counter < total/num
        
        for molname1 in train_mol_df.molecule_name:
            mol_df1 = atom_df.loc[(atom_df.molecule_name == molname1)]["typestr"]
            for molname2 in test_mol_df.molecule_name:
                mol_df2 = atom_df.loc[(atom_df.molecule_name == molname2)]["typestr"]
                count1 = list(mol_df1.values).count("N")
                count2 = list(mol_df2.values).count("N")
                assert count1 >= count2

# Pick max O to lowest O
def test_scheme_D4():
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
    
        chosen = select_molecules_D4(mol_df, atom_df, pair_df, prev_chosen=chosen, num=num)
        train_graphs, train_mol_df, test_graphs, test_mol_df = get_split(chosen, mol_df, graphs)

        assert len(train_graphs) == num * counter
        assert len(test_graphs) == total - (num * counter)
        assert len(train_mol_df.molecule_name.unique()) == len(train_mol_df.molecule_name)
        assert len(test_mol_df.molecule_name.unique()) == len(test_mol_df.molecule_name)
        assert counter < total/num
        
        for molname1 in train_mol_df.molecule_name:
            mol_df1 = atom_df.loc[(atom_df.molecule_name == molname1)]["typestr"]
            for molname2 in test_mol_df.molecule_name:
                mol_df2 = atom_df.loc[(atom_df.molecule_name == molname2)]["typestr"]
                count1 =  list(mol_df1.values).count("O")
                count2 =  list(mol_df2.values).count("O")
                assert count1 >= count2
                
# Pick max F to lowest F
def test_scheme_D5():
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
    
        chosen = select_molecules_D5(mol_df, atom_df, pair_df, prev_chosen=chosen, num=num)
        train_graphs, train_mol_df, test_graphs, test_mol_df = get_split(chosen, mol_df, graphs)

        assert len(train_graphs) == num * counter
        assert len(test_graphs) == total - (num * counter)
        assert len(train_mol_df.molecule_name.unique()) == len(train_mol_df.molecule_name)
        assert len(test_mol_df.molecule_name.unique()) == len(test_mol_df.molecule_name)
        assert counter < total/num
        
        for molname1 in train_mol_df.molecule_name:
            mol_df1 = atom_df.loc[(atom_df.molecule_name == molname1)]["typestr"]
            for molname2 in test_mol_df.molecule_name:
                mol_df2 = atom_df.loc[(atom_df.molecule_name == molname2)]["typestr"]
                count1 =  list(mol_df1.values).count("F")
                count2 =  list(mol_df2.values).count("F")
                assert count1 >= count2
                
# Pick max X to lowest X, where X!=[H, C, N, O, F]
def test_scheme_D6():
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
    
        chosen = select_molecules_D6(mol_df, atom_df, pair_df, prev_chosen=chosen, num=num)
        train_graphs, train_mol_df, test_graphs, test_mol_df = get_split(chosen, mol_df, graphs)

        assert len(train_graphs) == num * counter
        assert len(test_graphs) == total - (num * counter)
        assert len(train_mol_df.molecule_name.unique()) == len(train_mol_df.molecule_name)
        assert len(test_mol_df.molecule_name.unique()) == len(test_mol_df.molecule_name)
        assert counter < total/num
        
        for molname1 in train_mol_df.molecule_name:
            mol_df1 = atom_df.loc[(atom_df.molecule_name == molname1)]["typestr"]
            count1 = 0
            for type in mol_df1.values:
                if type not in ['H', 'C', 'N', 'O', 'F']:
                    count1 += 1
            for molname2 in test_mol_df.molecule_name:
                mol_df2 = atom_df.loc[(atom_df.molecule_name == molname2)]["typestr"]
                count2 = 0
                for type in mol_df2.values:
                    if type not in ['H', 'C', 'N', 'O', 'F']:
                        count2 += 1
                assert count1 >= count2


