import random
import numpy
import os
from torch import manual_seed, cuda, backends

def seed_setting(seed):
    manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed) # if use multi-GPU
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)