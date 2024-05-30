from models import *
import torch
import torch.optim as optim
import numpy as np
from data_handling import get_data
import argparse
from ray import tune,train
from functools import partial
def train_(args, split,config):
    data = get_data(args.dataset, split)

    best_eval_acc = 0
    best_eval_loss = 1e5
    bad_counter = 0
    best_test_acc = 0

    nout = 5

    # model = G2_GNN(data.num_node_features, args.nhid, nout, args.nlayers, args.GNN, args.G2_exp, args.drop_in, args.drop,
    #                args.use_G2_conv).to(args.device)

    model = G2_GNN(data.num_node_features, config["hid"], nout, config["nlayers"], args.GNN, config["G2_exp"], config["drop_in"], config["dropout"],
                   args.use_G2_conv).to(args.device)

    lf = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])

    @torch.no_grad()
    def test(model, data):
        model.eval()
        logits, accs, losses = model(data), [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            loss = lf(logits[mask], data.y.squeeze()[mask])
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
            losses.append(loss.item())
        return accs, losses

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.to(args.device))
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
        loss.backward()
        optimizer.step()

        [train_acc, val_acc, test_acc], [train_loss, val_loss, test_loss] = test(model, data)

        if args.use_val_acc == True:
            if (val_acc > best_eval_acc):
                best_eval_acc = val_acc
                best_test_acc = test_acc
                bad_counter = 0
            else:
                bad_counter += 1

        else:
            if (val_loss < best_eval_loss):
                best_eval_loss = val_loss
                best_test_acc = test_acc
            else:
                bad_counter += 1

        # if ((epoch+1) == args.patience):
        #     break
        if bad_counter>=args.patience:
            break
        log = 'Split: {:01d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(split, epoch, train_acc, val_acc, test_acc))
        print("patience",bad_counter)
        print("best_val_acc",best_eval_acc)
        print("best_test_acc",best_test_acc)
    return best_test_acc,best_eval_acc



def train1(args, split):
    data = get_data(args.dataset, split)

    best_eval_acc = 0
    best_eval_loss = 1e5
    bad_counter = 0
    best_test_acc = 0

    nout = 5

    model = G2_GNN(data.num_node_features, args.nhid, nout, args.nlayers, args.GNN, args.G2_exp, args.drop_in, args.drop,
                   args.use_G2_conv).to(args.device)

    lf = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    @torch.no_grad()
    def test(model, data):
        model.eval()
        logits, accs, losses = model(data), [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            loss = lf(logits[mask], data.y.squeeze()[mask])
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
            losses.append(loss.item())
        return accs, losses

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.to(args.device))
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
        loss.backward()
        optimizer.step()

        [train_acc, val_acc, test_acc], [train_loss, val_loss, test_loss] = test(model, data)

        if args.use_val_acc == True:
            if (val_acc > best_eval_acc):
                best_eval_acc = val_acc
                best_test_acc = test_acc
                bad_counter = 0
            else:
                bad_counter += 1

        else:
            if (val_loss < best_eval_loss):
                best_eval_loss = val_loss
                best_test_acc = test_acc
            else:
                bad_counter += 1

        # if ((epoch+1) == args.patience):
        #     break
        if bad_counter>=args.patience:
            break
        log = 'Split: {:01d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(split, epoch, train_acc, val_acc, test_acc))
        print("patience",bad_counter)
        print("best_val_acc",best_eval_acc)
        print("best_test_acc",best_test_acc)
    return best_test_acc












def run_splits(config,args):
    n_splits = 1
    best_results = []
    best_val_results = []
    for split in range(n_splits-1,n_splits):
        #best_results.append(train_(args, split,config))
        best_test_acc,best_val_acc = train_(args,split,config)
        best_results.append(best_test_acc)
        best_val_results.append(best_val_acc)
    best_results = np.array(best_results)
    mean_acc = np.mean(best_results)
    best_val_results = np.array(best_val_results)
    mean_val_acc = np.mean(best_val_results)
    std = np.std(best_results)
    metric = {"acc_val":mean_val_acc}
    train.report(metric)
    log = 'Final test results -- mean: {:.4f}, std: {:.4f}'
    print(log.format(mean_acc,std))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--dataset', type=str, default='wisconsin',
                        help='dataset name: texas, wisconsin, film, squirrel, chameleon, cornell')
    parser.add_argument('--GNN', type=str, default='GraphSAGE',
                        help='base GNN model used with G^2: GraphSAGE, GCN, GAT')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden node features')
    parser.add_argument('--nlayers', type=int, default=6,
                        help='number of layers')
    parser.add_argument('--epochs', type=int, default=500,
                        help='max epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--drop_in', type=float, default=0.5,
                        help='input dropout rate')
    parser.add_argument('--drop', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight_decay')
    parser.add_argument('--G2_exp', type=float, default=2.5,
                        help='exponent p in G^2')
    parser.add_argument('--use_val_acc', type=bool, default=True,
                        help='use validation accuracy for early stoppping -- otherwise use validation loss')
    parser.add_argument('--use_G2_conv', type=bool, default=True,
                        help='use a different GNN model for the gradient gating method')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='computing device')
    args = parser.parse_args()

    # n_splits = 2
    # best_results = []
    # for split in range(1,n_splits):
    #     best_results.append(train1(args, split))
    #     #tune.report(mean_accuracy=best_results[split])
    # best_results = np.array(best_results)
    # mean_acc = np.mean(best_results)
    # std = np.std(best_results)

    # log = 'Final test results -- mean: {:.4f}, std: {:.4f}'
    # print(log.format(mean_acc,std))



    config = {
    "lr":tune.loguniform(10e-4,1e-2),
    "hid":tune.choice([32,64,128,256,512]),
    "drop_in":tune.uniform(0,0.9),
    "dropout":tune.uniform(0,0.9),
    "weight_decay":tune.loguniform(1e-8,1e-2),
    "G2_exp":tune.uniform(1,5),
    #"nlayers":tune.randint(lower=5,upper=31) # 5 to 30
    "nlayers":tune.choice([5,10,15])
    }
    result = tune.run(partial(run_splits,args=args),config=config,resources_per_trial={"cpu": 1, "gpu": 1},metric="acc_val",
    mode="max",num_samples=20)
    best_trial = result.get_best_trial("acc_val", "max", "last")
    print(f"Best trial config: {best_trial.config}")