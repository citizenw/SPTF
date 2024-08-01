import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + "/../.."))
import torch
from src import launch_training

torch.set_num_threads(3)  # aviod high cpu avg usage


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-c", "--cfg", default="models/SPTF/STREETS.py", help="training config")
    parser.add_argument("-g", "--gpus", default="0", help="visible gpus")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    launch_training(args.cfg, args.gpus)
