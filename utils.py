import json
import os
import torch
import numpy as np
import random
from datetime import datetime
import logging


def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_saved_path(path):
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_path = path + '/' + name
    os.makedirs(saved_path, exist_ok=True)
    handlers = [logging.FileHandler(f'{saved_path}/log.txt'), logging.StreamHandler()]
    logging.basicConfig(level=logging.CRITICAL, format='%(message)s', handlers=handlers)
    return saved_path
