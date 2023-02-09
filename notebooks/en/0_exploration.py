# -*- coding: utf-8 -*-
import os
from tqdm.notebook import tqdm
from pathlib import Path
import pandas as pd
from nltk.lm import Vocabulary
import sys
sys.path.append("../../lib/post_ocr_correction")
from metrics import levenshtein

import pickle
folder = "../../data/ICDAR2019_POCR_competition_dataset/ICDAR2019_POCR_competition_training_18M_without_Finnish/EN/"

output_folder = Path("../../data/en")
files = sorted(os.listdir(folder))
len(files)

import glob

files = glob.glob(folder + '/**/*.txt', recursive=True)
len(files)

from multiprocessing import Pool


def extract(name):
    with open(name) as file:
        return file.readlines()


def create_windows(x):
    A, B, window_length = x
    assert len(A) == len(B)
    return [(A[i:i + window_length], B[i:i + window_length])
            for i in range(len(A) + 1)]


p = Pool(4)

data = list(p.imap_unordered(extract, tqdm(files), chunksize=128))
len(data)

# data = []
# for f in tqdm(files):
#     with open(f) as file:
#         data.append(file.readlines())

data = pd.DataFrame(data,
                    columns = ["ocr_to_input",
                               "ocr_aligned",
                               "gs_aligned"])\
.assign(ocr_to_input = lambda df: df.ocr_to_input.str.replace("[OCR_toInput] ", "", regex = False),
        ocr_aligned = lambda df: df.ocr_aligned.str.replace("[OCR_aligned] ", "", regex = False),
        gs_aligned = lambda df: df.gs_aligned.str.replace("[ GS_aligned] ", "", regex = False))

print(data.shape)
data.head()

data.applymap(len).describe()

levenshtein(reference = data.gs_aligned.str.replace("@", ""),
            hypothesis = data.ocr_to_input).cer.describe()

levenshtein(reference = data.gs_aligned,
            hypothesis = data.ocr_aligned).cer.describe()

vocabulary = Vocabulary(data.ocr_to_input.sum() + data.ocr_aligned.sum() + data.gs_aligned.sum())
print(len(vocabulary))
data_dir = output_folder/"data"
if not data_dir.exists():
    data_dir.mkdir(mode=755,exist_ok=True)
with open(data_dir/"vocabulary.pkl", "wb") as file:
    pickle.dump(vocabulary, file)

dev = data.sample(n = 5, random_state = 1)
dev.to_pickle(output_folder/"data/dev.pkl")
dev.shape

train = data.drop(dev.index)
train.to_pickle(output_folder/"data/train.pkl")
train.shape

train.applymap(len).describe()
dev.applymap(len).describe()
levenshtein(reference = dev.gs_aligned.str.replace("@", ""),
            hypothesis = dev.ocr_to_input).cer.describe()

levenshtein(reference = dev.gs_aligned,
            hypothesis = dev.ocr_to_input).cer.describe()
window_length = 100

df = train#.head(100)
train_aligned = list(p.imap_unordered(create_windows,
                                      tqdm(zip(df.ocr_aligned,
                                               df.gs_aligned,
                                               [window_length for x in df.ocr_aligned]),
                                           total = len(df.ocr_aligned)),
                                      chunksize = 128))
s = []
for r in tqdm(train_aligned):
    s.extend(r)
train_aligned = pd.DataFrame(s, columns = ["source", "target"])
print(train_aligned.shape)
train_aligned.head()
train_aligned = train_aligned.assign(source = lambda df: df.source.str.replace("@", ""))
train_aligned.head()

dev_aligned = dev.apply(lambda r: create_windows((r["ocr_aligned"], r["gs_aligned"], window_length)),
                            axis = 1).sum()
dev_aligned = pd.DataFrame(dev_aligned, columns = ["source", "target"])
print(dev_aligned.shape)
dev_aligned.head()

dev_aligned = dev_aligned.assign(source = lambda df: df.source.str.replace("@", ""))
dev_aligned.head()
train_aligned.to_pickle(output_folder/"data/train_aligned.pkl")
dev_aligned.to_pickle(output_folder/"data/dev_aligned.pkl")