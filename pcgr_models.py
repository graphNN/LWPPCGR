import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GraphConv, GINConv, APPNPConv, SAGEConv, SGConv

#import sys
#print(sys.path)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout, activation, bn):
        super(MLP, self).__init__()
        self.nl = num_layers
        self.drops = dropout
        self.acti = activation
        self.bn = bn

        self.bnlayer = nn.ModuleList()
        if bn:
            for i in range(num_layers):
                if i == 0:
                    self.bnlayer.append(nn.BatchNorm1d(input_dim))
                if 0 < i < num_layers:
                    self.bnlayer.append(nn.BatchNorm1d(hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            if 0 < i < num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.init_parameter()

    def init_parameter(self):
        for i in range(self.nl):
            stdv = 1. / math.sqrt(self.layers[i].weight.size(1))
            self.layers[i].weight.data.normal_(-stdv, stdv)

    def forward(self, x):
        h = x
        if self.bn:
            h = self.bnlayer[0](h)
        for i in range(self.nl):
            h = self.layers[i](h)
            if i < self.nl - 1:
                if self.acti:
                    h = F.relu(h)
                    # h = F.leaky_relu(h)
                if self.bn and i+1 < self.nl:
                    h = self.bnlayer[i+1](h)
                h = F.dropout(h, self.drops[1], self.training)

        return h


class GCNprop(nn.Module):
    def __init__(self, input_dim, hidden, classes, num_layer, dropout, activation):
        super(GCNprop, self).__init__()
        self.num_layer = num_layer
        self.layer = nn.ModuleList()
        self.drop = dropout
        self.acti = activation

        for i in range(num_layer):
            in_feat = input_dim if i == 0 else hidden
            out_feat = hidden if i < num_layer-1 else classes
            self.layer.append(GraphConv(in_feats=in_feat, out_feats=out_feat, allow_zero_in_degree=True, bias=False))

    def forward(self, g, feats):
        x = feats
        h = F.dropout(x, self.drop[0], self.training)

        for j in range(self.num_layer):
            h = self.layer[j](g, h)
            if j < self.num_layer - 1:
                if self.acti:
                    h = F.relu(h)
                h = F.dropout(h, self.drop[1], self.training)

        return h


class SAGEprop(nn.Module):
    def __init__(self, input_dim, hidden, classes, num_layer, aggregator_type, dropout, activation):
        super(SAGEprop, self).__init__()
        self.num_layer = num_layer
        self.layer = nn.ModuleList()
        self.drop = dropout
        self.acti = activation

        for i in range(num_layer):
            in_feat = input_dim if i == 0 else hidden
            out_feat = hidden if i < num_layer-1 else classes
            self.layer.append(SAGEConv(in_feats=in_feat, out_feats=out_feat, aggregator_type=aggregator_type))

    def forward(self, g, feats):
        x = feats
        h = F.dropout(x, self.drop[0], self.training)

        for j in range(self.num_layer):
            h = self.layer[j](g, h)
            if j < self.num_layer - 1:
                if self.acti:
                    h = F.relu(h)
                h = F.dropout(h, self.drop[1], self.training)

        return h


class SGCprop(nn.Module):
    def __init__(self, input_dim, hidden, classes, num_layer, dropout, activation):
        super(SGCprop, self).__init__()
        self.num_layer = num_layer
        self.layer = nn.ModuleList()
        self.drop = dropout
        self.acti = activation

        for i in range(num_layer):
            in_feat = input_dim if i == 0 else hidden
            out_feat = hidden if i < num_layer-1 else classes
            self.layer.append(SGConv(in_feats=in_feat, out_feats=out_feat, k=2))

    def forward(self, g, feats):
        x = feats
        h = F.dropout(x, self.drop[0], self.training)

        for j in range(self.num_layer):
            h = self.layer[j](g, h)
            if j < self.num_layer - 1:
                if self.acti:
                    h = F.relu(h)
                h = F.dropout(h, self.drop[1], self.training)

        return h

class GINprop(nn.Module):
    def __init__(self, input_dim, hidden, classes, num_prop, aggregator_type, dropout, activation):
        super(GINprop, self).__init__()
        self.num_prop = num_prop
        self.acti = activation
        self.drops = dropout
        self.apply_func = nn.ModuleList()
        self.layer = nn.ModuleList()

        for i in range(num_prop):
            if i == 0:
                self.apply_func.append(nn.Linear(input_dim, hidden))
            if 0 < i < num_prop - 1:
                self.apply_func.append(nn.Linear(hidden, hidden))
            if i == num_prop - 1:
                self.apply_func.append(nn.Linear(hidden, classes))

        for i in range(num_prop):
            self.layer.append(GINConv(apply_func=self.apply_func[i], aggregator_type=aggregator_type))

        self.set_parameters()


    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for i in range(self.num_prop):
            nn.init.xavier_normal_(self.apply_func[i].weight, gain=gain)

    def forward(self, g, feats):
        h = feats
        if self.drops[0] > 0:
            h = F.dropout(h, self.drops[0], self.training)

        for j in range(self.num_prop):
            h = self.layer[j](g, h)
            if j < self.num_prop - 1:
                if self.acti:
                    h = F.relu(h)
                h = F.dropout(h, self.drops[1], self.training)

        return h


class lwpcgr(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_prop, drops, hop, path, prop_type, aggregator_type
                 , activation, dataname):
        super(lwpcgr, self).__init__()
        self.hop = hop
        self.path = path
        self.prop_type = prop_type
        self.dataname = dataname
        self.hlw_lin = nn.Linear(1, num_prop)  # seed change
        hlw = torch.Tensor(hop)
        self.hlw = nn.Parameter(hlw)
        if prop_type == 'GCN':
            self.prop = GCNprop(input_dim, hidden_dim, out_dim, 2, drops, activation)
        elif prop_type == 'GIN':
            self.prop = GINprop(input_dim, hidden_dim, out_dim, num_prop, aggregator_type, drops, activation)
        elif prop_type == 'SAGE':
            self.prop = SAGEprop(input_dim, hidden_dim, out_dim, num_prop, aggregator_type, drops, activation)
        elif prop_type == 'SGC':
            self.prop = SGCprop(input_dim, hidden_dim, out_dim, num_prop, drops, activation)
        else:
            print('For models not yet designed, readers are invited to design their own, '
                  'note that our method supports any model!')

        self.init_parameter()  # ...

    def init_parameter(self):
        if self.dataname == 'chameleon':
            self.hlw.data.fill_(0.1)
        else:
            nn.init.normal_(self.hlw)

    def forward(self, graphs, feats):
        h = []
        h.append(self.prop(graphs[0], self.hlw[0] * feats))
        for i in range(self.hop - 1):
            x = self.prop(graphs[i + 1], self.hlw[i + 1] / (i + 2) * feats)
            h.append(x)
        z = sum(h)

        return z


class lwpcgr_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_prop, drops, hop, path, prop_type, aggregator_type
                 , activation, dataname):
        super(lwpcgr_1, self).__init__()
        self.hop = hop
        self.path = path
        self.prop_type = prop_type
        self.dataname = dataname
        self.hlw_lin = nn.Linear(1, num_prop)  # seed change
        hlw = torch.Tensor(hop)
        self.hlw = nn.Parameter(hlw)
        if prop_type == 'GCN':
            self.prop = GCNprop(input_dim, hidden_dim, out_dim, 2, drops, activation)
        elif prop_type == 'GIN':
            self.prop = GINprop(input_dim, hidden_dim, out_dim, num_prop, aggregator_type, drops, activation)
        elif prop_type == 'SAGE':
            self.prop = SAGEprop(input_dim, hidden_dim, out_dim, num_prop, aggregator_type, drops, activation)
        elif prop_type == 'SGC':
            self.prop = SGCprop(input_dim, hidden_dim, out_dim, num_prop, drops, activation)
        else:
            print('For models not yet designed, readers are invited to design their own, '
                  'note that our method supports any model!')

        self.init_parameter()  # ...

    def init_parameter(self):
        if self.dataname == 'chameleon':
            self.hlw.data.fill_(0.1)
        else:
            nn.init.normal_(self.hlw)

    def forward(self, graphs, feats):
        h = []
        h.append(self.prop(graphs[0], feats))

        z = sum(h)

        return z

class lwpcgr_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_prop, drops, hop, path, prop_type, aggregator_type
                 , activation, dataname):
        super(lwpcgr_2, self).__init__()
        self.hop = hop
        self.path = path
        self.prop_type = prop_type
        self.dataname = dataname
        self.hlw_lin = nn.Linear(1, num_prop)  # seed change
        hlw = torch.Tensor(hop)
        self.hlw = nn.Parameter(hlw)
        if prop_type == 'GCN':
            self.prop = GCNprop(input_dim, hidden_dim, out_dim, 2, drops, activation)
        elif prop_type == 'GIN':
            self.prop = GINprop(input_dim, hidden_dim, out_dim, num_prop, aggregator_type, drops, activation)
        elif prop_type == 'SAGE':
            self.prop = SAGEprop(input_dim, hidden_dim, out_dim, num_prop, aggregator_type, drops, activation)
        elif prop_type == 'SGC':
            self.prop = SGCprop(input_dim, hidden_dim, out_dim, num_prop, drops, activation)
        else:
            print('For models not yet designed, readers are invited to design their own, '
                  'note that our method supports any model!')

        self.init_parameter()  # ...

    def init_parameter(self):
        if self.dataname == 'chameleon':
            self.hlw.data.fill_(0.1)
        else:
            nn.init.normal_(self.hlw)

    def forward(self, graphs, feats):
        h = []
        h.append(self.prop(graphs[0], feats))
        for i in range(self.hop - 1):
            x = self.prop(graphs[i + 1], feats)
            h.append(x)
        z = sum(h)

        return z


class lwpcgr_3(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_prop, drops, hop, path, prop_type, aggregator_type
                 , activation, dataname):
        super(lwpcgr_3, self).__init__()
        self.hop = hop
        self.path = path
        self.prop_type = prop_type
        self.dataname = dataname
        self.hlw_lin = nn.Linear(1, num_prop)  # seed change
        hlw = torch.Tensor(hop)
        self.hlw = nn.Parameter(hlw)
        if prop_type == 'GCN':
            self.prop = GCNprop(input_dim, hidden_dim, out_dim, 2, drops, activation)
        elif prop_type == 'GIN':
            self.prop = GINprop(input_dim, hidden_dim, out_dim, num_prop, aggregator_type, drops, activation)
        elif prop_type == 'SAGE':
            self.prop = SAGEprop(input_dim, hidden_dim, out_dim, num_prop, aggregator_type, drops, activation)
        elif prop_type == 'SGC':
            self.prop = SGCprop(input_dim, hidden_dim, out_dim, num_prop, drops, activation)
        else:
            print('For models not yet designed, readers are invited to design their own, '
                  'note that our method supports any model!')

        self.init_parameter()  # ...

    def init_parameter(self):
        if self.dataname == 'chameleon':
            self.hlw.data.fill_(0.1)
        else:
            nn.init.normal_(self.hlw)

    def forward(self, graphs, feats):
        h = []
        h.append(self.prop(graphs[0], self.hlw[0] * feats))

        z = sum(h)

        return z

