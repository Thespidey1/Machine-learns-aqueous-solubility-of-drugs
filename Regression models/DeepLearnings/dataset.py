from __future__ import print_function, division
import functools
import  numpy  as  np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import functools
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset
import functools
from joblib import Parallel, delayed
from tqdm import tqdm


class SMILES_dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer):
        self.smiles = df['Normalized SMILES']
        
        # Remove tqdm progress bar, keep parallel tokenization
        self.tokens = np.array(
            list(Parallel(n_jobs=3)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in self.smiles
            ))
        )
        self.label = df['LogS(mol/L)']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)
    

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Tokenize the SMILES string to get X (shape should be [100, 1] if padding length is 100)
        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
        
        # Ensure y has shape (51,) where each value corresponds to a sigma_i
        y = torch.from_numpy(np.asarray(self.label[index])).float()  # Shape should be [51]

        smiles = self.smiles[index]
        # print(y)
        # print(X.shape, y.shape, smiles)

        return (X, y), y, smiles


import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Batch, Data
from descriptor import atom_features, bond_features, etype_features
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import math


def combine_Graph(Graph_list):
    """
    merge a Graph with multiple subgraph
    Args:
        Graph_list: list() of torch_geometric.data.Data object

    Returns: torch_geometric.data.Data object

    """
    x = Batch.from_data_list(Graph_list).x
    edge_index = Batch.from_data_list(Graph_list).edge_index
    edge_attr = Batch.from_data_list(Graph_list).edge_attr

    combined_Graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return combined_Graph


def add_global(graph):
    """
    add a global point, all the attribute are set to zero
    :param graph: pyg.data
    :return: pyg.data
    """
    node = torch.zeros(1, 40)
    # node.shape
    x = torch.cat([graph.x, node], dim=0)
    num_node = x.shape[0] - 1
    new_node = x.shape[0] - 1
    start = []
    end = []
    attr = []

    for i in range(num_node):
        # print(i)
        start.append(new_node)
        end.append(i)
        attr.append([0] * 10)

    start = torch.tensor(start).reshape(1, -1)
    end = torch.tensor(end).reshape(1, -1)
    new_edge = torch.cat([start, end], dim=0)
    edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    attr = torch.tensor(attr)
    edge_attr = torch.cat([graph.edge_attr, attr], dim=0)
    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return g


class Graph_dataset(torch.utils.data.Dataset):
    """
    torch dataset
    """

    def __init__(self, df):
        super(Graph_dataset, self).__init__()
        self.smiles = df['Normalized SMILES']
        self.label = df['LogS(mol/L)']
        self.index = df['index']
        self.length = len(df)
        # self.target = df['LogS(mol/L)']
        self.df = df

        # # show basic information
        # print("----info----")
        # print("data_length", self.length)
        # print("------------")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        smiles = row['Normalized SMILES']
        # cation_smiles = row['Normalized SMILES'].split('.')[0]
        # anion_smiles = row['Normalized SMILES'].split('.')[1]
        index = self.index
        # cation = self.smiles2graph(cation_smiles)
        # anion = self.smiles2graph(anion_smiles)
        # combine_graph = combine_Graph([cation, anion])
        # num_bond = combine_graph.edge_index.shape[1]
        combine_graph = self.smiles2graph(smiles)

        # combine_graph = add_global(combine_graph)
        label = torch.tensor(row['LogS(mol/L)'], dtype=torch.float).view(1, -1)

        data = Data(x=combine_graph.x, edge_index=combine_graph.edge_index, edge_attr=combine_graph.edge_attr, y=label,
                    index=index, mol_num=index, smiles=smiles)

        return data, label,smiles

    def smiles2graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        nodes = []
        edges = []
        edge_attrs = []

        for atom in mol.GetAtoms():
            node_feat = atom_features(atom)
            nodes.append(node_feat)

        for bond in mol.GetBonds():
            bond_feat = bond_features(bond)
            etype_feat = etype_features(bond)
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_attrs.append(etype_feat + bond_feat)

        x = np.array(nodes, dtype=np.float32)
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(edge_attrs, dtype=np.float32)

        x = torch.from_numpy(x)
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(edge_attr)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def collate_fn(self, batch):
        graphs = [data[0] for data in batch]
        batch = torch.cat([data.batch for data in graphs], dim=0)

        return batch