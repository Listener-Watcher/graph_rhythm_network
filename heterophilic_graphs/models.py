import torch
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,SimpleConv
from torch_geometric.utils import to_dense_adj

device = 'cuda:0'
class G2(nn.Module):
    def __init__(self, conv, nhid,p=2., conv_type='GraphSAGE', activation=nn.ReLU()):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type
        self.Q = nn.Linear(nhid*2,nhid)
    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        # gg = torch.tanh((scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
        #                          edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        gg = torch.tanh((scatter((torch.abs(self.Q(torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))) ** self.p).squeeze(-1),
                                  edge_index[0], 0,dim_size=X.size(0), reduce='mean')))

        return gg

class G2_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GraphSAGE', p=2., drop_in=0, drop=0, use_gg_conv=True):
        super(G2_GNN, self).__init__()
        self.conv_type = conv_type
        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = nlayers
        self.num_heads = 1
        self.p = p
        self.Q = nn.Linear(nhid,nhid)
        self.Q_list = nn.ModuleList()
        self.Q_list.append(nn.Linear(nhid*2,nhid))
        for i in range(self.num_heads):
            self.Q_list.append(nn.Linear(nhid,nhid))
        if conv_type == 'GCN':
            self.conv = GCNConv(nhid, nhid)
            #self.conv = SimpleConv()
            if use_gg_conv == True:
                self.conv_gg = GCNConv(nhid, nhid)
                #self.conv_gg = SimpleConv()
        elif conv_type == 'GraphSAGE':
            self.conv = SAGEConv(nhid, nhid)
            if use_gg_conv == True:
                self.conv_gg = SAGEConv(nhid, nhid)
                #self.conv_gg2 = SAGEConv(nhid, nhid)
        elif conv_type == 'GAT':
            self.conv = GATConv(nhid,nhid,heads=4,concat=True)
            if use_gg_conv == True:
                self.conv_gg = GATConv(nhid,nhid,heads=4,concat=True)
        else:
            print('specified graph conv not implemented')

        if use_gg_conv == True:
            self.G2 = G2(self.conv_gg,nhid,p,conv_type,activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv,p,nhid,conv_type,activation=nn.ReLU())

    def forward(self, data):
        X = data.x
        n_nodes = X.size(0)
        edge_index = data.edge_index
        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))
        #print(self.conv_type)
        # total = self.nlayers+1
        # total_prob = torch.ones((n_nodes,X.shape[1])).to("cuda:0")
        # a_last = total_prob/total
        X_stored = X
        for i in range(self.nlayers):
            if self.conv_type == 'GAT':
                X_ = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X_ = torch.relu(self.conv(X, edge_index))
            tau = self.G2(X, edge_index)
            # # X_2 = obtain_parameter_free_neighbor_info(X_,edge_index)
            # X_merge = torch.cat((X_,X_2),dim=1)
            # QX = torch.abs(self.Q_list[0](X_merge))
            # for j in range(1,self.num_heads):
            #     QX = torch.abs((self.Q_list[j](QX)))
            # QX = torch.abs(QX)
            # QX = torch.pow(QX,self.p)
            # tau = torch.tanh(QX)
            # tau = tau*(torch.div(a_last,(1-(total-2)*a_last)))
            #tau = torch.rand(X.shape).to(device)
            X = (1 - tau) * X + tau * X_
            # if i%1==0:
            #     #X = (1 - tau) * X + tau * X_
            #     tau = self.G2(X, edge_index)
            #     X = (1-tau)*X_stored+tau*X_
            #     X_stored = X
            # else:
            #     X = X_
            #X = X_
            # node_st = torch.zeros((n_nodes,5))
            # energy = self.G2(X,edge_index)
            # for j in range(n_nodes):
            #     if torch.mean(energy[j])<0.01:
            #         node_st[j][0]+=1
            #     elif torch.mean(energy[j])<0.1:
            #         node_st[j][1]+=1
            #     elif torch.mean(energy[j])<0.5:
            #         node_st[j][2]+=1
            #     elif torch.mean(energy[j])<0.9:
            #         node_st[j][3]+=1
            #     else:
            #         node_st[j][4]+=1
            # print("statistics:",torch.sum(node_st,axis=0))

            # total_prob = total_prob-torch.div(tau,(torch.div(a_last,(1-(total-2)*a_last))))*a_last
            # total-=1
            # a_last = total_prob/total
        X = F.dropout(X, self.drop, training=self.training)

        return self.dec(X)
