# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from nltk.lm import Vocabulary
import torch
import torch.utils.data as tud
import sys
sys.path.append("../../lib/post_ocr_correction/")
from metrics import levenshtein
import pickle
from pathlib import Path
import re
folder = Path("../../data/en/data/")
train = pd.read_pickle(folder/"train_aligned.pkl")
train.shape