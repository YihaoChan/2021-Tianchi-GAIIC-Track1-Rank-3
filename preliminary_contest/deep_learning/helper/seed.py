import random
import os
import torch
import numpy as np
from torch.backends import cudnn


def seed():
    seed_num = 7

    random.seed(seed_num)

    os.environ['PYTHONHASHSEED'] = str(seed_num)

    np.random.seed(seed_num)

    torch.manual_seed(seed_num)

    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    cudnn.enabled = False
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.cuda.empty_cache()
