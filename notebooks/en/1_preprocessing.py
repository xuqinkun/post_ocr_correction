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
print(train.shape)

dev = pd.read_pickle(folder/"dev_aligned.pkl")
print(dev.shape)

with open(folder/"vocabulary.pkl", "rb") as file:
    vocabulary = pickle.load(file)

char2i = {c:i for i, c in enumerate(sorted(vocabulary), 3)}
char2i["<PAD>"] = 0
char2i["<START>"] = 1
char2i["<END>"] = 2
print(len(char2i))

i2char = {i:c for i, c in enumerate(sorted(vocabulary), 3)}
i2char[0] = "<PAD>"
i2char[1] = "<START>"
i2char[2] = "<END>"
print(len(i2char))

length = 100

output = []
for s in tqdm(train.source):
    output.append(torch.tensor([1] + [char2i[c] for c in s] + [2]))

train_source = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
print(train_source.shape)

output = []
for s in tqdm(train.target):
    output.append(torch.tensor([1] + [char2i[c] for c in s] + [2]))

train_target = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
print(train_target.shape)

train.source[0] == re.sub(r"<START>|<END>|<PAD>", "", "".join([i2char[c] for c in train_source[0].tolist()]))

train.target[0] == re.sub(r"<START>|<END>|<PAD>", "", "".join([i2char[c] for c in train_target[0].tolist()]))

output = []
for s in tqdm(dev.source):
    output.append(torch.tensor([1] + [char2i[c] for c in s] + [2]))

dev_source = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
print(dev_source.shape)

output = []
for s in tqdm(dev.target):
    output.append(torch.tensor([1] + [char2i[c] for c in s] + [2]))

dev_target = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
print(dev_target.shape)

dev.source[0] == re.sub(r"<START>|<END>|<PAD>", "", "".join([i2char[c] for c in dev_source[0].tolist()]))
dev.target[0] == re.sub(r"<START>|<END>|<PAD>", "", "".join([i2char[c] for c in dev_target[0].tolist()]))

torch.save(train_source, folder/"train_source.pt")
torch.save(train_target, folder/"train_target.pt")
torch.save(dev_source, folder/"dev_source.pt")
torch.save(dev_target, folder/"dev_target.pt")
with open(folder/"char2i.pkl", "wb") as file:
    pickle.dump(char2i, file)
with open(folder/"i2char.pkl", "wb") as file:
    pickle.dump(i2char, file)