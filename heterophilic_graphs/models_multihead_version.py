import torch
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,SimpleConv
from torch_geometric.utils import to_dense_adj,dropout_edge,dense_to_sparse
import syslog
device = 'cuda:0'
class G2(nn.Module):
    def __init__(self, conv, nhid,p=2., conv_type='GraphSAGE',heads=1, activation=nn.ReLU()):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type
        self.Q = nn.Linear(nhid*2,nhid)
        self.Q_list = nn.ModuleList()
        self.heads = heads
        for i in range(self.heads):
            self.Q_list.append(nn.Linear(nhid*2,int(nhid/self.heads)))
    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        #gg2 = torch.tanh((scatter((torch.abs(self.Q(torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))) ** self.p).squeeze(-1),
        #                          edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        #gg  = torch.tanh((scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        #print("energy",torch.sum(energy))
        #gg = torch.tanh((scatter((torch.abs(self.Q(torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))+self.Q(torch.cat((X[edge_index[1]],X[edge_index[0]]),dim=1))) ** self.p).squeeze(-1),edge_index[0],0,dim_size=X.size(0), reduce='mean')))
        #Degree = torch.sum(X,dim=0) 
        #gg = torch.tanh((scatter((torch.abs(Degree[edge_index[0]]-Degree[edge_index[1]])**self.p),edge_index[0],0,dim_size=X.size(0),reduce='mean')))
	#gg2 = torch.tanh((scatter((torch.abs(self.Q2(torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))) ** self.p).squeeze(-1),
        #                          edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        #-------------------------------
        #print("one head dimension",self.Q_list[0](torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1)).squeeze(-1).shape)
        #print(torch.cat([(torch.abs(self.Q_list[i](torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))) ** self.p).squeeze(-1) for i in range(self.heads)],dim=1).shape)
        gg = torch.tanh((scatter(torch.cat([(torch.abs(self.Q_list[i](torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))) ** self.p).squeeze(-1) for i in range(self.heads)],dim=1),
                                edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        #print("tau shape",gg.shape)
        gg_avg = gg
        return gg_avg
    def log_energy(self, X, edge_index):
        n_nodes = X.size(0)
        #gg2 = torch.tanh((scatter((torch.abs(self.Q(torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))) ** self.p).squeeze(-1),
        #                          edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        # gg  = torch.tanh((scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
        #                           edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        #print("energy",torch.sum(energy))
        #gg = torch.tanh((scatter((torch.abs(self.Q(torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))+self.Q(torch.cat((X[edge_index[1]],X[edge_index[0]]),dim=1))) ** self.p).squeeze(-1),edge_index[0],0,dim_size=X.size(0), reduce='mean')))
        #Degree = torch.sum(X,dim=0) 
        #gg = torch.tanh((scatter((torch.abs(Degree[edge_index[0]]-Degree[edge_index[1]])**self.p),edge_index[0],0,dim_size=X.size(0),reduce='mean')))
    #gg2 = torch.tanh((scatter((torch.abs(self.Q2(torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))) ** self.p).squeeze(-1),
        #                          edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        #-------------------------------
        gg = torch.tanh((scatter((torch.abs(self.Q(torch.cat((X[edge_index[0]],X[edge_index[1]]),dim=1))) ** self.p).squeeze(-1),
                                edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        gg2 = torch.tanh((scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
        return gg,gg2

class G2_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GraphSAGE', p=2., drop_in=0, drop=0, use_gg_conv=True,beta=1,heads=1):
        super(G2_GNN, self).__init__()
        self.conv_type = conv_type
        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = nlayers
        self.num_heads = heads
        self.p = p
        self.beta = beta
        # self.bn = nn.BatchNorm1d(nhid)
        #self.Q = nn.Linear(nhid,nhid)
        #self.Q_list = nn.ModuleList()
        #self.Q_list.append(nn.Linear(nhid*2,nhid))
        #for i in range(self.num_heads):
        #    self.Q_list.append(nn.Linear(nhid,nhid))
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
            self.G2 = G2(self.conv_gg,nhid,p,conv_type,self.num_heads,activation=nn.ReLU())
            #self.G3 = G2(self.conv_gg2,nhid,p,conv_type,activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv,p,nhid,conv_type,self.num_heads,activation=nn.ReLU())

    def log_energy(self, data):
        X = data.x
        n_nodes = X.size(0)
        edge_index = data.edge_index

        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))

        for i in range(self.nlayers):
            if self.conv_type == 'GAT':
                X_ = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
                #X_2 = F.elu(self.conv(X,edge_index2)).view(n_nodes,-1,4).mean(dim=-1)
            else:
                X_ = torch.relu(self.conv(X, edge_index))
                #X_2 = torch.relu(self.conv(X,edge_index2))
            tau = self.G2(X, edge_index)
            gg,gg2 = self.G2.log_energy(X,edge_index)
            X = (1-tau)*X+tau*X_

        gg,gg2 = self.G2.log_energy(X,edge_index)
        with open('energy_list.txt', 'w') as f:
            torch.set_printoptions(edgeitems=torch.inf)
            print('learned energy:',gg[0], file=f)
            print('Dirichlet energy:', gg2[0],file=f)
        X = F.dropout(X, self.drop, training=self.training)
        return self.dec(X)

    def forward(self, data):
        X = data.x
        n_nodes = X.size(0)
        edge_index = data.edge_index
        #print("edge_index",edge_index)
        #print("adj",to_dense_adj(edge_index))
        #A = to_dense_adj(edge_index)
        #A2 =  (torch.matmul(A,A)>0).int()
        #edge_index2 = dense_to_sparse(A2)
        #edge_index,_ = dropout_edge(edge_index,p=self.drop_in,training=self.training)
        #A = to_dense_adj(edge_index)
        #A2 = ((torch.matmul(A,A))>0).int()
        #edge_index2,_ = dense_to_sparse(A2)
        #print("???",edge_index2)
        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))
        #print(self.conv_type)
        # total = self.nlayers+1
        # total_prob = torch.ones((n_nodes,X.shape[1])).to("cuda:0")
        # a_last = total_prob/total
        #X_stored = X
        for i in range(self.nlayers):
            if self.conv_type == 'GAT':
                X_ = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
                #X_2 = F.elu(self.conv(X,edge_index2)).view(n_nodes,-1,4).mean(dim=-1)
            else:
                m = torch.nn.ReLU()
                X_ = m(self.conv(X, edge_index))
                #X_2 = torch.relu(self.conv(X,edge_index2))
            tau = self.G2(X, edge_index)
            #tau2 = self.G3(X,edge_index2)
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
            
            #Tau = torch.stack((tau,tau2),dim=0)
            #print(Tau.shape)
            #Tau = F.softmax(Tau,dim=0)
            #print(Tau[0][0]+Tau[1][0])
            #print("tau",Tau[0].shape)
            #X = self.beta*((1-tau) * X + tau * X_)+(1-self.beta)*((1-tau2)*X+tau2*X_2)
            #X = (1-tau)*X+tau*X_
            X = (1-tau)*X+tau*X_
			#X = X_
            #print("tau",Tau[1][0][0])
            #print("tau",Tau[2][0][0])
            #X = Tau[0]*X+Tau[1]*X_+Tau[2]*X_2
            #X = X_
            #energy  = torch.tanh((scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** 2).squeeze(-1),edge_index[0], 0,dim_size=X.size(0), reduce='mean')))
            #energy = torch.sum(energy)
            #f = open("energy_list.txt","w")
            #with open('energy_list.txt','a') as f:
            #if i %5 ==0:
            #    print("layer:"+str(i))
            #    print(energy)
            #f.close()
            #X = X_

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
        #with open('energy_list.txt','a') as f:
        #    print("end of one training",file = f)
        X = F.dropout(X, self.drop, training=self.training)
        return self.dec(X)
