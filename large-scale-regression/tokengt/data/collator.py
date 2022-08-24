"""
Modified from https://github.com/microsoft/Graphormer
"""

import torch
import torch.nn.functional as F


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


@torch.no_grad()
def collator(
        items,
        max_node=512,
        max_edge=2048,
        multi_hop_max_dist=20,
        spatial_pos_max=20
):
    items = [item for item in items if
             item is not None and item.x.size(0) <= max_node and item.edge_attr.size(0) <= max_edge]
    
    # print("<-- ITEMS -->")
    # print(items)
    # print("<-- ITEMS -->\n")

    (
        idxs,
        edge_index,
        edge_data,
        node_data,
        in_degree,
        out_degree,
        lap_eigvec,
        lap_eigval,
        ys,
        ring_node,
        ring_node_lap_eigvec,
        ring_node_lap_eigval
    ) = zip(*[
        (
            item.idx,
            item.edge_index,
            item.edge_data,
            item.node_data,
            item.in_degree,
            item.out_degree,
            item.lap_eigvec,
            item.lap_eigval,
            item.y,
            item.ring_node,
            item.ring_node_lap_eigvec,
            item.ring_node_lap_eigval
        )
        for item in items
    ])

    #NOTE: node, edge num
    node_num = [i.size(0) for i in node_data]
    edge_num = [i.size(0) for i in edge_data]
    max_n = max(node_num)

    ring_num = [len(i) for i in ring_node]
    
    # print("COLLATOR")
    # print("node_num")
    # print(len(node_num))
    # print(node_num)
    # print("node_data")
    # print(len(node_data))
    # print(node_data)
    # print("ring_node")
    # print(len(ring_node))
    # print(ring_node)

    #NOTE: Calcualte ring feature
    ring_data = []
    for idx, ring_set in enumerate(ring_node):
        new_ft_ls = []
        # print(ring_set)
        for ring_nodes in ring_set:
            # print(ring_nodes)
            new_ft = sum([node_data[idx][node-1]//len(ring_nodes) for node in ring_nodes])
            new_ft_ls.append(new_ft)

        if len(ring_set) !=0:
            new_ring_feat = torch.stack(new_ft_ls)
        else:
            new_ring_feat = torch.zeros(1, 9, dtype=int)
        # print(new_ring_feat)

        ring_data.append(new_ring_feat)
    ring_data = tuple(ring_data)

    y = torch.cat(ys)  # [B,]
    edge_index = torch.cat(edge_index, dim=1)  # [2, sum(edge_num)]
    edge_data = torch.cat(edge_data) + 1  # [sum(edge_num), De], +1 for nn.Embedding with pad_index=0
    node_data = torch.cat(node_data) + 1  # [sum(node_num), Dn], +1 for nn.Embedding with pad_index=0
    in_degree = torch.cat(in_degree) + 1  # [sum(node_num),], +1 for nn.Embedding with pad_index=0
    out_degree = torch.cat(out_degree) + 1  # [sum(node_num),], +1 for nn.Embedding with pad_index=0
    ring_data = torch.cat(ring_data) + 1

    # [sum(node_num), Dl] = [sum(node_num), max_n]
    lap_eigvec = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigvec])
    lap_eigval = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigval])

    result = dict(
        idx=torch.LongTensor(idxs),
        edge_index=edge_index,
        edge_data=edge_data,
        node_data=node_data,
        in_degree=in_degree,
        out_degree=out_degree,
        lap_eigvec=lap_eigvec,
        lap_eigval=lap_eigval,
        y=y,
        node_num=node_num,
        edge_num=edge_num,
        ring_num=ring_num,
        ring_data=ring_data,
        ring_node=ring_node,
        ring_node_lap_eigvec=ring_node_lap_eigvec,
        ring_node_lap_eigval=ring_node_lap_eigval
    )

    return result
