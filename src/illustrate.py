import torch
from torch.optim import SGD, Rprop, RMSprop
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from model import LinearRegression, LogisticRegression
from data import *
from viz import *

def init_args():
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, help="Epoch", default=100)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-2)
    parser.add_argument("-m", type=float, help="Initial slope", default=0.0)
    parser.add_argument("-c", type=float, help="Initial intercept", default=0.5)
    parser.add_argument("--cls", action="store_true", help="Classification problem")
    parser.add_argument("--data", type=str, help="Data path")
    parser.add_argument("--out", type=str, help="Output file name", default="./output.mp4")
    args = parser.parse_args()

    return args

def train(optimizer_class, dataloader, args, **kwargs):
    model = LogisticRegression(m=args.m, c=args.c) if args.cls else LinearRegression(m=args.m, c=args.c)

    criterion = torch.nn.BCELoss(reduction="none") if args.cls else torch.nn.MSELoss(reduction="none")
    optimizer = optimizer_class(model.parameters(), lr=args.lr, **kwargs)

    history_loss = []
    history_slope = []
    history_intercept = []

    for e in range(args.epoch):
        all_loss = []
        all_slope = []
        all_intercept = []
        for x, y in dataloader:
            optimizer.zero_grad()
            x, y = x.float(), y.float()
            o = model.forward(x)

            mini_batch_loss = criterion(o, y)
            
            all_loss.extend(mini_batch_loss.view(-1).tolist())
            all_slope.extend([model.m.item() for n in range(len(o))])
            all_intercept.extend([model.c.item() for n in range(len(o))])
            
            mini_batch_loss = mini_batch_loss.mean()

            mini_batch_loss.backward()
            optimizer.step()
        history_loss.append(all_loss)
        history_slope.append(all_slope)
        history_intercept.append(all_intercept)
    return {
        "loss" : history_loss,
        "slope" : history_slope,
        "intercept" : history_intercept
    }

def main():
    args = init_args()

    if args.data:
        dataloader = load_data(args.data)
    else:
        if args.cls:
            dataloader = load_data_cls()
        else:
            dataloader = load_data_reg()
    
    history_sgd = train(SGD, dataloader, args)
    history_rprop = train(Rprop, dataloader, args, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    history_rmsprop = train(RMSprop, dataloader, args, alpha=0.8)

    all_slope = np.array(history_sgd["slope"]).flatten().tolist() + np.array(history_rprop["slope"]).flatten().tolist() + np.array(history_rmsprop["slope"]).flatten().tolist()
    all_intercept = np.array(history_sgd["intercept"]).flatten().tolist() + np.array(history_rprop["intercept"]).flatten().tolist() + np.array(history_rmsprop["intercept"]).flatten().tolist()

    slope_range = (min(all_slope), max(all_slope))
    intercept_range = (min(all_intercept), max(all_intercept))

    plt.ioff()
    gradient_frames = animate_gradient(0, dataloader, args, slope_range, intercept_range, history_sgd, history_rprop, history_rmsprop)
    fit_frames = animate_fit(0, dataloader, args, history_sgd, history_rprop, history_rmsprop)
    frames = [combine_images(gradient_frames[i], fit_frames[i]) for i in range(args.epoch)]

    create_video_from_images(frames, args.out, 10)

if __name__ == "__main__":
    main()