from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor,Planetoid, KarateClub
import torch
import numpy as np
# from visualize2 import *
def get_data(name, split=0):
	print("data handle")
	path = './data/' +name
	if name in ['chameleon','squirrel']:
		dataset = WikipediaNetwork(root=path, name=name)
	if name in ['cornell', 'texas', 'wisconsin']:
		dataset = WebKB(path ,name=name)
	if name == 'film':
		dataset = Actor(root=path)
	if name == 'cora':
		dataset = Planetoid(root=path,name=name,split='public')
	if name == 'Citeseer':
		dataset = Planetoid(root=path,name=name,split='public')
	if name == 'Pubmed':
		dataset = Planetoid(root=path,name=name,split='public')
	if name == 'karate':
		dataset = KarateClub()
	data = dataset[0]
	#print(data.train_mask)
	#print(data.test_mask)
	#print(data.val_mask)
	if name in ['chameleon', 'squirrel']:
		splits_file = np.load(f'{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')
	if name in ['cornell', 'texas', 'wisconsin']:
		splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
	if name == 'film':
		splits_file = np.load(f'{path}/raw/{name}_split_0.6_0.2_{split}.npz')
	#if name == 'cora':
	#	splits_file = np.load(f'{path}/{name}/geom-gcn/raw/{name}_split_0.6_0.2_{split}.npz')
	if name in ['chameleon','squirrel','cornell','texas','wisconsin','film']:
		train_mask = splits_file['train_mask']
		val_mask = splits_file['val_mask']
		test_mask = splits_file['test_mask']
		data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
		data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
		data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
	return data
# data = get_data(name='cora')
# X = data.x
# A = data.edge_index
# F = compute_frequency(X,A)
# F_mean = torch.mean(F,dim=1)
# x_axis = torch.arange(F_mean.shape[0])
# Y_list = [F_mean]
# tag_list = ["frequency_mean"]
# visualize_signal(x_axis,Y_list,tag_list)


