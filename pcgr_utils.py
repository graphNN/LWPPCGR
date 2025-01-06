import dgl
import networkx
import torch
import scipy.sparse as sp
import numpy as np
import copy

import os
import sys
import pickle as pkl
#import networkx as nx
import random
import networkx as nx
#from networkx.algorithms import approximation


def pcgr_(adj, H, P):  # adj: 0 or 1
    adjs = []
    adjs.append(adj)
    adj_ah = copy.deepcopy(adj)
    adj_drop = copy.deepcopy(adj+ torch.eye(adj.shape[0]))
    for i in range(1, H):
        adj_ah = torch.matmul(adj_ah, adj)
        adj_h = torch.where(adj_drop == 0, adj_ah, 0)
        # adjs.append(adj_h)
        adj_hp = torch.where(adj_h > P, 0, adj_h)
        adjs.append(adj_hp)
        adj_drop = copy.deepcopy(adj_ah) + adj_drop
    return adjs


def pcgr_pre(adj, H, P):  # adj: 0 or 1  #
    #adjh = []
    adjp = []
    #adjh.append(sp.csr_matrix(adj))
    adj_ah = copy.deepcopy(adj)
    adj_drop = copy.deepcopy(adj)
    for h in range(2, H+1):
        adj_ah = torch.matmul(adj_ah, adj)
        adj_h = torch.where(adj_drop == 0, adj_ah, 0)
        #adjh.append(sp.csr_matrix(adj_h))
        for p in range(1, P+1):
            adj_hp = torch.where(adj_h == p, adj_h, 0)  # Go save memory.
            adjp.append(sp.csr_matrix(adj_hp))
            print('h and p:', h, p)
        adj_drop = adj_ah + adj_drop

    return adjp


#@jit(nopython=True)
def pcgr_large_preprocess(g, H, P):
    gsp = []
    nxg = dgl.to_networkx(g)
    nodes = g.nodes()

    for p in range(0, P*(H-1)):
        gsp.append(dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=len(nodes)))
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if nx.has_path(nxg, i, j):
                h = nx.shortest_path_length(nxg, i, j)
                if 2 <= h <= H:
                    p = len(list(nx.all_shortest_paths(nxg, i, j)))
                    if 0 < p <= P:
                        gsp[P * (h - 2) + p - 1].add_edges(torch.tensor(i), torch.tensor(j))

        if i % 1000 == 0:
           print(i)
    transform = dgl.AddReverse()
    #transform1 = dgl.AddSelfLoop()
    transform2 = dgl.ToSimple()
    for i in range(len(gsp)):
        gsp[i] = transform(gsp[i])
        #gsp[i] = transform1(gsp[i])
        gsp[i] = transform2(gsp[i])
        gsp[i].remove_edges((torch.tensor([0]), torch.tensor([0])))

    return gsp


def pcgr_large_pre_(g, H, P):
    gsh = []
    gsp = []
    gsh.append(g)
    nxg = dgl.to_networkx(g)
    shortest_path = dict(networkx.all_pairs_shortest_path_length(nxg))
    #path_num = approximation.all_pairs_node_connectivity(nxg)
    shortest_path_num = dict(networkx.all_pairs_all_shortest_paths(nxg))

    nodes = g.nodes()
    for h in range(1, H):
        gsh.append(dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=len(nodes)))
    for p in range(0, P*(H-1)):
        gsp.append(dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=len(nodes)))
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if j in shortest_path[i] and 2 <= shortest_path[i][j] <= H:
                h = shortest_path[i][j]
                gsh[h-1].add_edges(torch.tensor([i]), torch.tensor([j]))
                if 0 < len(shortest_path_num[i][j]) <= P:
                    p = len(shortest_path_num[i][j])
                    for m in range(p, P+1):
                        gsp[P * (h - 2) + m - 1].add_edges(torch.tensor(i), torch.tensor(j))

    transform = dgl.AddReverse()
    transform1 = dgl.AddSelfLoop()
    transform2 = dgl.ToSimple()
    for i in range(len(gsh)):
        gsh[i] = transform(gsh[i])
        gsh[i] = transform1(gsh[i])
        gsh[i] = transform2(gsh[i])
    for i in range(len(gsp)):
        gsp[i] = transform(gsp[i])
        gsp[i] = transform1(gsp[i])
        gsp[i] = transform2(gsp[i])

    return [gsh, gsp]


def pcgr_large_pre(g, H, P):
    gsh = []
    gsp = []
    gsh.append(g)
    nxg = dgl.to_networkx(g)
    shortest_path = dict(networkx.all_pairs_shortest_path_length(nxg))
    shortest_path_num = dict(networkx.all_pairs_all_shortest_paths(nxg))

    nodes = g.nodes()
    for h in range(1, H):
        graph = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=len(nodes))
        for p in range(1, P+1):
            graphp = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=len(nodes))
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if j in shortest_path[i] and shortest_path[i][j] == h + 1:
                        graph.add_edges(torch.tensor([i]), torch.tensor([j]))
                        if 0 < len(shortest_path_num[i][j]) <= p:
                            graphp.add_edges(torch.tensor(i), torch.tensor(j))
            print('h and p:', h, p)
            gsp.append(graphp)
        gsh.append(graph)

    transform = dgl.AddReverse()
    transform1 = dgl.AddSelfLoop()
    transform2 = dgl.ToSimple()
    for i in range(len(gsh)):
        gsh[i] = transform(gsh[i])
        gsh[i] = transform1(gsh[i])
        gsh[i] = transform2(gsh[i])
    for i in range(len(gsp)):
        gsp[i] = transform(gsp[i])
        gsp[i] = transform1(gsp[i])
        gsp[i] = transform2(gsp[i])

    return [gsh, gsp]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def norm_adj(adj):
    D1 = (np.array(adj.sum(axis=1)) ** (-0.5))
    D2 = (np.array(adj.sum(axis=0)) ** (-0.5))

    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)

    A = sparse_mx_to_torch_sparse_tensor(A)

    return A


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def consis_loss(logps, tem, lam):
    logps = torch.exp(logps)
    sharp_logps = (torch.pow(logps, 1. / tem) / torch.sum(torch.pow(logps, 1. / tem), dim=1, keepdim=True)).detach()
    loss = torch.mean((logps - sharp_logps).pow(2).sum(1)) * lam

    return loss


test_seeds = [
    2406525885, 3164031153, 1454191016, 1583215992, 765984986,
    258270452, 3808600642, 292690791, 2492579272, 1660347731,
    902096533, 1295255868, 3887601419, 2250799892, 4099160157,
    658822373, 1105377040, 1822472846, 2360402805, 2355749367,
    2291281609, 1241963358, 3431144533, 623424053, 78533721,
    1819244826, 1368272433, 555336705, 1979924085, 1064200250,
    256355991, 125892661, 4214462414, 2173868563, 629150633,
    525931699, 3859280724, 1633334170, 1881852583, 2776477614,
    1576005390, 2488832372, 2518362830, 2535216825, 333285849,
    109709634, 2287562222, 3519650116, 3997158861, 3939456016,
    4049817465, 2056937834, 4198936517, 1928038128, 897197605,
    3241375559, 3379824712, 3094687001, 80894711, 1598990667,
    2733558549, 2514977904, 3551930474, 2501047343, 2838870928,
    2323804206, 2609476842, 1941488137, 1647800118, 1544748364,
    983997847, 1907884813, 1261931583, 4094088262, 536998751,
    3788863109, 4023022221, 3116173213, 4019585660, 3278901850,
    3321752075, 2108550661, 2354669019, 3317723962, 1915553117,
    1464389813, 1648766618, 3423813613, 1338906396, 629014539,
    3330934799, 3295065306, 3212139042, 3653474276, 1078114430,
    2424918363, 3316305951, 2059234307, 1805510917, 1327514671
]

val_seeds = [4258031807, 3829679737, 3706579387, 789594926,  3628091752]
development_seed = 1684992425

def set_train_val_test_split(
        seed: int,
        labels,
        dataset_name: str = 'Cora',
        num_development: int = 1500,
        num_per_class: int = 20):
    y = labels
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        for c in range(y.max() + 1):
            class_idx = development_idx[np.where(y[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))
    else:
        num_train = int(0.6*num_nodes)
        train_idx.extend(rnd_state.choice(development_idx, num_train, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    train_mask = get_mask(train_idx)
    val_mask = get_mask(val_idx)
    test_mask = get_mask(test_idx)

    return train_mask, val_mask, test_mask


def random_splits(labels, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,labels.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=len(labels))
    val_mask = index_to_mask(val_idx,size=len(labels))
    test_mask = index_to_mask(test_idx,size=len(labels))
    return train_mask, val_mask, test_mask


def fixed_splits(labels, num_classes, percls_trn, val_lb, name, seed=42):
    if name in ["Chameleon","Squirrel", "Actor"]:
        seed = 1941488137
    index=[i for i in range(0,labels.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=len(labels))
    val_mask = index_to_mask(val_idx,size=len(labels))
    test_mask = index_to_mask(test_idx,size=len(labels))

    return train_mask, val_mask, test_mask


file_dir_citation = os.getcwd() + '/data'
def load_data_citation(dataset_str='cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(file_dir_citation, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(file_dir_citation, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features_norm = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    D1 = (np.array(adj.sum(axis=1)) ** (-0.5))[np.newaxis, :]
    D2 = (np.array(adj.sum(axis=0)) ** (-0.5))[np.newaxis, :]
    D1 = sp.diags(D1[0], format='csr')
    D2 = sp.diags(D2[0], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    # features = torch.FloatTensor(np.array(features.todense()))
    features_norm = torch.FloatTensor(np.array(features_norm.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, features_norm, labels, idx_train, idx_val, idx_test, adj


from load_geom import load_geom
from dgl.data import CoauthorCSDataset, CoraFullDataset, CoauthorPhysicsDataset
SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539,
             3212139042, 2424918363]
def load_data(dataset, full, random_split, train_rate, val_rate, i):
    if dataset in {'cora', 'citeseer', 'pubmed'}:
        A, features, labels, train_mask, val_mask, test_mask, adj = load_data_citation(dataset)
        g = dgl.from_scipy(sp.coo_matrix(adj))
        g = dgl.remove_self_loop(g)
        adj = g.adj(scipy_fmt='csr')
        percls_trn = int(round(train_rate * len(labels) / int(labels.max() + 1)))
        val_lb = int(round(val_rate * len(labels)))
        if full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
    if dataset in {'texas', 'cornell', "chameleon", 'squirrel', 'wisconsin', 'film'}:
        path = os.getcwd()
        dataset_split = path + '/splits/' + f'{dataset}_split_0.6_0.2_{i}.npz'
        g, features, labels, train_mask, val_mask, test_mask = load_geom(dataset, dataset_split,
                                                                         train_percentage=None, val_percentage=None,
                                                                         embedding_mode=None, embedding_method=None,
                                                                         embedding_method_graph=None,
                                                                         embedding_method_space=None)
        percls_trn = int(round(train_rate * len(labels) / int(labels.max() + 1)))
        val_lb = int(round(val_rate * len(labels)))
        g = dgl.remove_self_loop(g)
        adj = g.adj(scipy_fmt='csr')
        if full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        if not full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        if not full and not random_split:
            train_mask, val_mask, test_mask = fixed_splits(labels, int(labels.max()) + 1, percls_trn, val_lb, dataset)
    if dataset in {'cora-full', 'cs', 'physics'}:
        if dataset == 'cora-full':
            data = CoraFullDataset()
        if dataset == 'cs':
            data = CoauthorCSDataset()
        if dataset == 'physics':
            data = CoauthorPhysicsDataset()

        g = data[0]  # .to(device)
        g = dgl.remove_self_loop(g)
        adj = g.adj(scipy_fmt='csr')
        features = g.ndata['feat']
        labels = g.ndata['label']

        percls_trn = int(round(train_rate * len(labels) / int(labels.max() + 1)))
        val_lb = int(round(val_rate * len(labels)))
        if full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        if not full and random_split:
            train_mask, val_mask, test_mask = random_splits(labels, int(labels.max()) + 1, percls_trn, val_lb,
                                                            seed=SEEDS[i])
        if not full and not random_split:
            train_mask, val_mask, test_mask = fixed_splits(labels, int(labels.max()) + 1, percls_trn, val_lb, dataset)

    return g, adj, features, labels, train_mask, val_mask, test_mask

