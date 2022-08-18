"""
Modified from https://github.com/microsoft/Graphormer
"""

import os
import os.path as osp
import shutil
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import networkx as nx
import itertools


class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        """
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        def is_cycle_edge(i1, i2, cycle):
            if i2 == i1 + 1:
                return True
            if i1 == 0 and i2 == len(cycle) - 1:
                return True
            return False

        def is_chordless(graph, cycle):
            for (i1, v1), (i2, v2) in itertools.combinations(enumerate(cycle), 2):
                if not is_cycle_edge(i1, i2, cycle) and graph.has_edge(v1, v2):
                    return False
            return True

        print('Converting SMILES strings into graphs...')
        data_list = []
        for idx, i in enumerate(tqdm(range(len(smiles_list)))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            #NOTE: nx graph
            # Using Code from CWN
            # source: cwn/data/helper_test.py
            nx_graph = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"])
            nx_cycles = sorted(nx.simple_cycles(nx_graph))
            rings_node = set()
            rings_edge = set()
            for nx_cycle in nx_cycles:
                if len(nx_cycle) <= 2:
                    continue
                if not is_chordless(nx_graph, nx_cycle):
                    continue

                rings_node.add(tuple(sorted(nx_cycle)))
                #NOTE: Make use of chordless function, find edges in cycle
                edges = set()
                for (i1, v1), (i2, v2) in itertools.combinations(enumerate(nx_cycle), 2):
                    if is_cycle_edge(i1, i2, nx_cycle) and nx_graph.has_edge(v1, v2):
                        edges.add(tuple({v1, v2}))
                rings_edge.add(tuple(sorted(edges)))

            ring_cnt = len(rings_node)

            data.ring_cnt = int(ring_cnt)
            data.ring_node = sorted(rings_node)
            data.ring_edge = sorted(rings_edge)

            # NOTE: Cell Features?

            # TEST print
            if idx % 100000 == 0:
                # print(str(idx) + "th data")
                # print("<--Cycles Found-->")
                # print("Ring Cnt: " + str(data.ring_cnt))
                # print(data.ring_node)
                # print(data.ring_edge)
                # print("<--            -->\n")
            # TEST print

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


if __name__ == '__main__':
    dataset = PygPCQM4Mv2Dataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())
