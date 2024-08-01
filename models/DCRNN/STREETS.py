import os
import sys
import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from src.runners import SimpleTimeSeriesForecastingRunner
from src.data import TimeSeriesForecastingDataset
from src.losses import masked_mae
from src.utils import load_adj

from .arch import DCRNN

CFG = EasyDict()

# DCRNN does not allow to load parameters since it creates parameters in the first iteration
resume = False
if not resume:
    import random
    _ = random.randint(-1e6, 1e6)

# ================= general ================= #
CFG.DESCRIPTION = "DCRNN model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "STREETS"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG._ = _
CFG.GPU_NUM = 1
CFG.NULL_VAL = np.nan

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "DCRNN"
CFG.MODEL.ARCH = DCRNN
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME +
                     "/adj_mx.pkl", "doubletransition")
CFG.MODEL.PARAM = {
    "cl_decay_steps": 2000,
    "horizon": 12,
    "input_dim": 2,
    "max_diffusion_step": 2,
    "num_nodes": 322,
    "num_rnn_layers": 2,
    "output_dim": 1,
    "rnn_units": 64,
    "seq_len": 12,
    "adj_mx": [torch.tensor(i) for i in adj_mx],
    "use_curriculum_learning": True
}
CFG.MODEL.SETUP_GRAPH = True
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.003,
    "eps": 1e-3
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [80],
    "gamma": 0.3
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
