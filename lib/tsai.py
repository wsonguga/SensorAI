import torch
import numpy as np
# from utils import *
import os
from tsai.all import *
import random
import argparse
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

def trend_plot(pred, label, name):

    index = np.argsort(label)
    pred = pred[index]
    label = label[index]
    mae = np.mean(np.abs(pred - label))
    me = np.mean(pred - label)
    correlation_matrix = np.corrcoef(pred, label)
    correlation = correlation_matrix[0, 1]
    std = np.std(pred - label)
    xy = np.vstack([label, pred])
    density = gaussian_kde(xy)(xy)
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B3B3EB", "#3636FF", "#00006C"])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(np.arange(1, pred.shape[0]+1), pred, c=density, cmap=custom_cmap, s=1)
    plt.scatter(np.arange(1, pred.shape[0]+1), label, c="red", s=1)
    plt.colorbar(scatter, label='Density')
    plt.legend(["Prediction", "Label"], loc="upper left")
    plt.text(1, 0, f"MAE:{mae:6.2f} ME:  {me:6.2f}\nSTD: {std:6.2f} Corr: {correlation:3.2f}", fontsize="x-large", ha='right', va='bottom', transform=plt.gca().transAxes)
    # plt.title(f'Trend Plot of {name}')
    # plt.savefig(f"./fig/{name}_trend_plot.png")
    plt.show()


def seed_everything(seed):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Typical Training Process') 
    parser.add_argument('--model', type=str, default=None,
                        help='TSAI model')
    parser.add_argument('--show_graph', type=bool, default=False,
                        help='show training graph')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='batch size (default: 512)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='upper epoch limit (default: 500)')
    parser.add_argument('--seq_len', type=int, default=1000,
                        help='sequence length (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='run on which device')

    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if args.device == 'cuda:0':
        torch.cuda.set_device(0)
    elif args.device == 'cuda:1':
        torch.cuda.set_device(1)
    else:
        raise ValueError("Invalid device")

    data_train = np.load("../data/BSG_train_rr.npy")
    data_test = np.load("../data/BSG_test_rr_0.1.npy")
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)
    X_train = data_train[:,:1000]
    Y_train = data_train[:,-1]
    X_test = data_test[:,:1000]
    Y_test = data_test[:,-1]
    X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
    X_train = X_train / np.max(np.abs(X_train), axis=1, keepdims=True)
    X_test = X_test - np.mean(X_test, axis=1, keepdims=True)
    X_test = X_test / np.max(np.abs(X_test), axis=1, keepdims=True)
    print('X train shape:', X_train.shape, 'Y train shape:', Y_train.shape, 'X test shape:', X_test.shape, 'Y test shape:', Y_test.shape)

    tfms  = [None, [TSRegression()]]
    splits = get_splits(X_train, n_splits=1, shuffle=True, check_splits=True)
    dls = get_ts_dls(X_train, Y_train, splits = splits, tfms=tfms, bs=args.batch_size)
    dls.one_batch()

    MODEL_DICT = {
        "LSTM": LSTM,
        "GRU": GRU,
        "MLP": MLP,
        "FCN": FCN,
        "ResNet": ResNet,
        "LSTM_FCN": LSTM_FCN,
        "GRU_FCN": GRU_FCN,
        "mWDN": mWDN,
        "TCN": TCN,
        "MLSTM_FCN": MLSTM_FCN,
        "InceptionTime": InceptionTime, 
        "MiniRocket": MiniRocket,
        "XceptionTime": XceptionTime, 
        "ResCNN": ResCNN,
        "TabModel": TabModel, #
        "OmniScaleCNN": OmniScaleCNN,
        "TST": TST,
        "TabTransformer": TabTransformer, #
        "TSiT": TSiT, #
        "XCM": XCM,
        "gMLP": gMLP,
        "TSPerceiver": TSPerceiver, #
        "GatedTabTransformer": GatedTabTransformer, #
        "TSSequencerPlus": TSSequencerPlus, 
        "PatchTST": PatchTST #
    }
    model = MODEL_DICT.get(args.model, None)
    cbs_graph = []  # Empty callback list
    if args.show_graph:
        cbs_graph.append(ShowGraph())
    # print(device)
    learn3 = ts_learner(dls, model, metrics=[mae, rmse], cbs=cbs_graph, device=device)
    learn3.model.to(device)
    print(f"Model is now on: {next(learn3.model.parameters()).device}")
    os.makedirs('./pth', exist_ok=True)
    early_stop = EarlyStoppingCallback(monitor="valid_loss", patience=3)
    learn3.fit_one_cycle(args.epochs, cbs=[early_stop, SaveModelCallback(monitor="valid_loss", fname=f"{args.model}_best_model")])
    learn3.export(f"./pth/{args.model}.pkl")

    X_train = np.expand_dims(X_train, axis = -2)
    X_test = np.expand_dims(X_test, axis = -2)

    learn3 = load_learner(f"./pth/{args.model}.pkl", cpu=False)
    Predictions_train=learn3.get_X_preds(X_train)[0]
    Predictions_test=learn3.get_X_preds(X_test)[0]

    trend_plot(Predictions_train.cpu().detach().numpy().flatten(), Y_train.flatten(), f"{args.model}_train")
    trend_plot(Predictions_test.cpu().detach().numpy().flatten(), Y_test.flatten(), f"{args.model}_test")