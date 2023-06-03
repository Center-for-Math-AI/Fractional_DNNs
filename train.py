# train.py

import argparse, os, datetime, time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import utils
from utils import floatFormat, makedirs

parser = argparse.ArgumentParser('Integer DNN')
# dataset
parser.add_argument('--data', choices=['peaks', 'sine'], type=str, default='peaks')
parser.add_argument('--n_train', type=int, default=10000)
parser.add_argument('--n_test', type=int, default=1000)

# network details
parser.add_argument('--net', choices=['ResNN', 'fDNN'], type=str, default='fDNN')
parser.add_argument('--m', type=int, default=32, help="net width (neurons)")
parser.add_argument('--nTh', type=int, default=3, help="net depth (layers)")
parser.add_argument('--resume', type=str, default=None, help="for loading a pretrained model")
parser.add_argument('--learn_tau', default=True, help="whether to learn the step-size")

# optimizer details
parser.add_argument('--epoch', type=int, default=1500)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optim', type=str, default='adam', choices=['adam'])
parser.add_argument('--weight_decay', type=float, default=0.0)

# loss
parser.add_argument('--loss_fn', type=str, default='MSE', choices=['MSE', 'CrossEnt'])

# dirs and precision
parser.add_argument('--save', type=str, default='experiments/run', help="define the save directory")
parser.add_argument('--prec', type=str, default='single', choices=['single', 'double'],
                    help="single or double precision")

# log freq
parser.add_argument('--val_freq', type=int, default=3, help="how often to run model on validation set")
parser.add_argument('--print_freq', type=int, default=2, help="how often to print results to log")

args = parser.parse_args()

# set precision
if args.prec == 'double':
    argPrec = torch.float64
else:
    argPrec = torch.float32

###################################
# add timestamp to save path
sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# logger
logger = {}
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving=True)
logger.info("start time: " + sStartTime)
logger.info(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

if __name__ == "__main__":
    torch.set_default_dtype(argPrec)
    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    # load data
    n_train = int(0.9 * args.n_train)
    n_val = int(0.1 * args.n_train)
    n_test = args.n_test

    if args.data == "peaks":
        from get_datasets import peaks

        din, dout = 2, 1
        x = cvt(-3 + 6 * torch.rand(n_train + n_val + n_test, 2))
        y = peaks(x)

    elif args.data == "sine":
        from get_datasets import sinosoindal

        din, dout = 1, 1
        x = cvt(-3 + 6 * torch.rand(n_train + n_val + n_test,1))
        y = sinosoindal(x)

    # shuffle and split data
    idx = torch.randperm(n_train + n_val + n_test)
    x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
    x_val, y_val = x[idx[n_train:n_train + n_val]], y[idx[n_train:n_train + n_val]]
    x_test, y_test = x[idx[n_train + n_val:]], y[idx[n_train + n_val:]]



    # set-up model
    if args.resume is not None:
        logger.info(' ')
        logger.info("loading model: {:}".format(args.resume))
        logger.info(' ')

        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        args.m = checkpt['args'].m
        args.nTh = checkpt['args'].nTh

    if args.net == 'ResNN':
        from networks import ResNN

        net = ResNN(din=din, m=args.m, dout=dout, nTh=args.nTh, tau_learn = args.learn_tau)
    elif args.net == 'fDNN':
        from networks import fDNN

        net = fDNN(din=din, m=args.m, dout=dout, nTh=args.nTh, tau_learn = args.learn_tau)

    if args.resume is not None:
        net.load_state_dict(checkpt["state_dict"])

    net = net.to(argPrec).to(device)

    # Pytorch optimizer for the network weights
    if args.optim == 'adam':
        optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    strTitle = args.data + '_' + sStartTime + f'_width-{str(args.m)}_lr-{floatFormat(args.lr)}' + f'_depth-{str(args.nTh)}'

    # loss functional
    if args.loss_fn == "MSE":
        loss_fn = nn.MSELoss()
    elif args.loss_fn == "CrossEnt":
        loss_fn = nn.CrossEntropyLoss()

    logger.info("---------------------- Network ----------------------------")
    logger.info(net)
    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(net)))
    logger.info("--------------------------------------------------")
    logger.info(str(optim))  # optimizer info
    logger.info(str(loss_fn))  # loss info
    logger.info("data={:} device={:}".format(args.data, device))
    logger.info("epoch={:} val_freq={:}".format(args.epoch, args.val_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info(strTitle)
    logger.info("--------------------------------------------------\n")

    columns = ["epoch", "lr", "time", "loss", "trainloss", "valloss"]
    hist = pd.DataFrame(columns=columns)
    hist = hist.astype({'epoch':'int'})

    # initial evaluation
    train_loss = loss_fn(net(x_train), y_train)
    val_loss = loss_fn(net(x_val), y_val)

    # printing
    hist.loc[len(hist.index)] = [-1, 0, 0, 0, train_loss.item(), val_loss.item()]
    log_msg = (hist.astype({'epoch':'int'})).to_string(index=False)
    logger.info(log_msg)

    best_loss = float('inf')
    bestParams = None
    time_meter = utils.AverageMeter()
    end = time.time()

    # training
    for epoch in range(args.epoch):
        # training here
        net.train()
        n = x_train.shape[0]
        b = args.batch_size
        n_batch = n // b
        loss = torch.zeros(1)
        running_loss = 0.0

        # shuffle
        idx = torch.randperm(n)

        for i in range(n_batch):
            idxb = idx[i * b:(i + 1) * b]
            xb, yb = x_train[idxb], y_train[idxb]
            if args.optim == 'adam':
                optim.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                optim.step()

            running_loss += b * loss.item()
        time_meter.update(time.time() - end)
        running_loss = running_loss / n

        # test
        train_loss = loss_fn(net(x_train), y_train)

        # validation
        if epoch % args.val_freq == 0 or epoch == args.epoch:
            val_loss = loss_fn(net(x_val), y_val)
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                makedirs(args.save)
                bestParams = net.state_dict()
                torch.save({
                    'epoch': epoch,
                    'args': args,
                    'state_dict': bestParams,
                }, os.path.join(args.save, strTitle + '_checkpt.pth'))
            hist.loc[len(hist.index)] = [epoch, time_meter.val, args.lr, running_loss, train_loss.item(), val_loss.item()]
        else:
            hist.loc[len(hist.index)] = [epoch, time_meter.val, args.lr, running_loss, train_loss.item(), ""]

        # printing
        if epoch % args.print_freq == 0 or epoch == args.epoch:
            ch = hist.iloc[-1:]
            log_message = (ch.astype({'epoch':'int'})).to_string(index=False, header=False)
            logger.info(log_message)

        end = time.time()

    hist = hist.astype({'epoch': 'int'})
    hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info('Training has finished.  ' + os.path.join(args.save, strTitle))

    logger.info("-------------overall performance on test data for best model-------------------")
    net.load_state_dict(bestParams)
    test_loss = loss_fn(net(x_test), y_test)
    logger.info('Test Loss: %0.4e' % test_loss.item())