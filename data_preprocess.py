from pcgr_utils import pcgr_, sparse_mx_to_torch_sparse_tensor
import os
import dgl
import torch
import pickle
import numpy as np
import scipy.sparse as sp
from pcgr_utils import load_data, pcgr_large_pre, pcgr_pre, pcgr_large_pre_, pcgr_large_preprocess


transform = dgl.AddReverse()
transform1 = dgl.AddSelfLoop()
transform2 = dgl.ToSimple()


for data in {'cora'}:  # 'pubmed', 'citeseer'

    g, adj, _, _, _, _, _ = load_data(data, True, True, 0.6, 0.2, 0)
    gg = transform(g)
    ggg = transform2(gg)
    # gs = pcgr_(sparse_mx_to_torch_sparse_tensor(ggg.adj(scipy_fmt='csr')).to_dense(), 3, 2)
    # for i in range(len(gs)):
    #     g = transform1(transform(dgl.from_scipy(sp.csr_matrix(gs[i]))))

    if data == 'pubmed':
        H = 5
        P = 10
    else:
        H = 10
        P = 10

    gs1 = (pcgr_pre(sparse_mx_to_torch_sparse_tensor(ggg.adj(scipy_fmt='csr')).to_dense(), H, P))
    #gs1 = pcgr_large_preprocess(g, H, P)
    gsh = []
    gsp = []
    print('save data!')

    torch.save([[], gs1], os.getcwd() + '/pcgr_graphs' + f'/{data}.pkl')

    print(f'{data} is done!')

