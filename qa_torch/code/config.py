import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
import utils


import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(323)


class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    EPOCHS = 5
    ROBERTA_PATH = "../input/roberta-base"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "../input/tweet-train-folds-v2/train_folds.csv"
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file=f"{ROBERTA_PATH}/vocab.json", 
        merges_file=f"{ROBERTA_PATH}/merges.txt", 
        lowercase=True,
        add_prefix_space=True
    )