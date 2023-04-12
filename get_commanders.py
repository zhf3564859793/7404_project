import argparse


def base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true",
                        help="forbid using cuda")

    # input and output dir
    parser.add_argument("--data_dir", default='./data/Decagon', type=str,
                        help="[Decagon | DeepDDI]")
    parser.add_argument("--data_prefix", default='decagon', type=str,
                        help="[ddi_constraint | decagon | ddi]")
    parser.add_argument("--seed", default=1, type=int,
                        help="random seed")
    parser.add_argument("--only_test", action="store_true",default=False,
                        help="random seed")
    parser.add_argument("--topk", default=10, type=int,
                        help="topk precision")
    parser.add_argument("--resume_path", type=str, default=None,
                        help="learning rate")

    # paramters
    parser.add_argument("--epoch", default=1500, type=int,
                        help="training epoch")
    parser.add_argument("--lr", default=0.005, type=float,
                        help="learning rate")
    parser.add_argument("--data_ratio", default=100, type=float,
                        help="percentage of edge num")
    parser.add_argument("--n_iter_no_change", default=65, type=float,
                        help="Maximum number of epochs to not meet ``tol`` improvement.")
    parser.add_argument("--train_size", default=0.8, type=float,
                        help="train_size")
    parser.add_argument("--val_size", default=0.9, type=float,
                        help="validation size")
    return parser


def base_commanders():
    parser = base_parser()

    args = parser.parse_args()
    return args


def deepwalk_commanders():
    parser = base_parser()
    parser.add_argument("--representation-size", default=50, type=int,
                        help="representation-size")

    args = parser.parse_args()
    return args


def energy_commanders():
    parser = base_parser()
    parser.add_argument("--no_ce_loss", action="store_true",default=False,
                        help="don't use ce_loss in loss_max")
    parser.add_argument("--energy_phi_weight", default=0.1, type=float,
                        help="energy_phi_weight")

    args = parser.parse_args()
    return args
