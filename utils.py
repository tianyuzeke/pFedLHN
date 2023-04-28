import sys
import os
import argparse, logging

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args(parser):
    parser.add_argument("--suffix", type=str, default="", help="suffix")
    parser.add_argument("--norm-var", type=float, default=0.002, help="")
    parser.add_argument("--n-embeds", type=int, default=5, help="")
    parser.add_argument("--embed-dim", type=int, default=40, help="embedding dim")
    parser.add_argument("--grad-clip", type=int, default=30, help="")
    parser.add_argument("--classes-per-node", type=int, default=0, help="")
    parser.add_argument("--random-seed", type=int, default=0, help="")
    parser.add_argument("--save-model", type=str2bool, default=False, help="")

    parser.add_argument("--run-test", type=str2bool, default=False, help="")
    parser.add_argument("--train-clients", type=int, default=-1, help="train first # clients")
    parser.add_argument("--test-last", type=int, default=-1, help="")
    parser.add_argument("--model-path", type=str, default="", help="")

    parser.add_argument("--layer-wise", type=str2bool, default=True, help="")
    parser.add_argument("--use-fc", type=str2bool, default=True, help="")
    parser.add_argument("--hdim", type=int, default=128, help="")

    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100', 'mnist'], help="dir path for dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=50, help="number of simulated nodes")

    parser.add_argument("--clients-per-round", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'adamw', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")

    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
   
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels for cnn model")

    parser.add_argument("--cuda", type=int, default=-1, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="results", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    return args

def mkdir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

def get_logger(name='default', filename='./log.txt', enable_console=True):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(message)s',
                        datefmt='[%m-%d %H:%M:%S]',
                        filename=filename,
                        filemode='w')
    if enable_console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s  %(message)s', datefmt='[%m-%d %H:%M:%S]')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
    return logging.getLogger(name)