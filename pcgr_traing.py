import torch
import dgl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import argparse
import tqdm
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from pcgr_utils import norm_adj, accuracy, set_seed, load_data, sparse_mx_to_torch_sparse_tensor, \
    pcgr_, set_train_val_test_split, pcgr_large_preprocess
from pcgr_models import lwpcgr

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)


def train(model, optimizer, gs, x, y, masks, scheduler):
    model.train()
    optimizer.zero_grad()
    output = model(gs, x)
    output = F.log_softmax(output, dim=1)
    loss = F.cross_entropy(output[masks[0]], y[masks[0]])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss)
    del output


def test(model, gs, x, y, masks):
    model.eval()
    output, accs, losses = model(gs, x), [], []
    logits = F.log_softmax(output, dim=1)
    for i in range(3):
        acc = accuracy(logits[masks[i]], y[masks[i]])
        loss = F.cross_entropy(logits[masks[i]], y[masks[i]])
        accs.append(acc)
        losses.append(loss)

    return accs, losses, logits


def run(args, dataname, gs, full, random_split, i):
    if args.random_split:
        set_seed(args.seed)
    else:
        set_seed(i)
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    _, _, features, labels, _, _, _ = load_data(dataname, full, random_split, args.train_rate,
                                                                        args.val_rate, i)

    if args.random_split:
        non_test, test_mask = train_test_split(g.nodes(), test_size=1 - 0.6 - 0.2, random_state=i)  # split method
        train_mask, val_mask = train_test_split(non_test, test_size=0.2 / (0.8), random_state=i)
    else:
        if dataname in ['Cora', 'Citeseer', 'Pubmed']:
            num_development = 1500
            train_mask, val_mask, test_mask = set_train_val_test_split(i, labels, dataname,
                                                                       num_development=num_development)
        else:
            num_development = int(0.8 * labels.shape[0])
            train_mask, val_mask, test_mask = set_train_val_test_split(i, labels, dataname,
                                                                       num_development=num_development)

    model = lwpcgr(features.shape[1], args.hidden, int(max(labels)) + 1, args.props, args.dropout,
                       args.hop, args.path, args.prop_type, args.aggregator_type, activation=True, dataname=args.dataset)

    optimizer = torch.optim.Adam(
            [{'params': model.prop.parameters(), 'weight_decay': args.wd_lin, 'lr': args.lr_lin},
             {'params': model.hlw, 'weight_decay': args.wd_hlw, 'lr': args.lr_hlw}])
    scheduler = ReduceLROnPlateau(optimizer, patience=25)

    model.to(device)
    features = features.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    train_mask = train_mask.clone().detach().to(device)
    val_mask = val_mask.clone().detach().to(device)
    test_mask = test_mask.clone().detach().to(device)
    masks = [train_mask, val_mask, test_mask]
    for i in range(len(gs)):
        gs[i] = gs[i].to(device)

    best_acc, best_val_acc, test_acc, best_val_loss, bad_epoch = 0, 0, 0, float("inf"), float("inf")
    train_losses = []
    val_losses = []
    run_time = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train(model, optimizer, gs, features, labels, masks, scheduler)
        run_time.append(time.time() - t0)
        [train_acc, val_acc, tmp_test_acc], [train_loss, val_loss, tmp_test_loss], logits = test(model, gs,
                                                                                                 features, labels,
                                                                                                 masks)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            bad_epoch = 0
            # hlw = model.hlw.data.cpu()
        else:
            bad_epoch += 1
        if bad_epoch == args.patience:
            break

    return test_acc, best_val_loss, run_time


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer',
                    help='texas, cornell, wisconsin, chameleon, squirrel, cora, citeseer, pubmed')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--device', type=int, default=0, help='GPU device.')
parser.add_argument('--runs', type=int, default=10, help='number of runs.')
parser.add_argument('--activation', type=bool, default=True)
parser.add_argument('--full', type=bool, default=True, help='Whether full-supervised')
parser.add_argument('--random_split', type=bool, default=True, help='Whether random split')
parser.add_argument('--large', type=bool, default=False, help='Whether random split')

#parser.add_argument('--ablation', type=bool, default=False)
#parser.add_argument('--ablation_type', type=str, help='lwpcgr, lwpcgr_no_add, lwpcgr_no_lw, lwpcgr_no_add_lw')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.') # 64/128
parser.add_argument('--props', type=int, default=2, help='The number of message passing layers')
parser.add_argument('--pro_dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability) of propagation.')
parser.add_argument('--lin_dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability) of linear.')
parser.add_argument('--hop', type=int, default=4, help='H')  # TEXAS CORNELL 10
parser.add_argument('--path', type=int, default=10, help='P')
parser.add_argument('--prop_type', type=str, default='SAGE', help="GCN, GIN, SAGE, SGC")
parser.add_argument('--aggregator_type', type=str, default='mean', help=", ")
parser.add_argument('--lr_lin', type=float, default=0.001, help='Initial learning rate of linear.')
parser.add_argument('--wd_lin', type=float, default=0.005, help='Weight decay (L2 loss on parameters) of linear.')
parser.add_argument('--lr_hlw', type=float, default=0.5, help='Initial learning rate of coefficients.')
parser.add_argument('--wd_hlw', type=float, default=0.005, help='Weight decay (L2 loss on parameters) of coefficients.')

args = parser.parse_args()
print(args)

args.dropout = [args.pro_dropout, args.lin_dropout]

if args.full:
    args.train_rate = 0.6
    args.val_rate = 0.2
else:
    args.train_rate = 0.025
    args.val_rate = 0.025

print(f'LWPCGR+{args.prop_type}: dataset loading and preprocessing...')
g, adj, _, _, _, _, _ = load_data(args.dataset, args.full, args.random_split,
                                      args.train_rate, args.val_rate, 0)
transform = dgl.AddReverse()
transform1 = dgl.ToSimple()
g = dgl.remove_self_loop(transform1(transform(g)))

if os.path.exists(os.getcwd() + '/pcgr_graphs' + f'/{args.dataset}.pkl'):
    adjs = torch.load(os.getcwd() + '/pcgr_graphs' + f'/{args.dataset}.pkl')
    gs = []
    gs.append(dgl.add_self_loop(g))
    for i in range(args.hop - 1):
        # gs.append(dgl.from_scipy(adjs[0][i + 1] + sp.eye(len(g.nodes()))))
        adj_ = []
        for p in range(1, args.path + 1):
            adj_.append(adjs[1][i * 10 + p - 1])
        gs.append(dgl.from_scipy(sum(adj_) + sp.eye(len(g.nodes()))))
else:
    if args.large:
        gs = []
        graphs = pcgr_large_preprocess(g, args.hop, args.path)
        gs.append(dgl.add_self_loop(g))
        for i in range(len(graphs)):
            gs.append(dgl.from_scipy(graphs[i] + sp.eye(len(g.nodes()))))
    else:
        gs = []
        adjs = pcgr_(sparse_mx_to_torch_sparse_tensor(g.adj(scipy_fmt='csr')).to_dense(), args.hop, args.path)
        for i in range(len(adjs)):
            gs.append(dgl.from_scipy(sp.coo_matrix(adjs[i])))
results = []
time_results = []
all_test_accs = []

for i in tqdm.tqdm(range(args.runs)):
    test_acc, best_val_loss, run_time = run(args, args.dataset, gs, args.full, args.random_split, i)
    time_results.append(run_time)
    all_test_accs.append(test_acc.item())  #
    print(f'run_{str(i + 1)} \t test_acc: {test_acc:.4f}')
run_sum = 0
epochsss = 0
for i in time_results:
    run_sum += sum(i)
    epochsss += len(i)

print("each run avg_time:", run_sum / (args.runs), "s")
print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")
print('test acc mean (%) =', np.mean(all_test_accs) * 100, np.std(all_test_accs) * 100)
