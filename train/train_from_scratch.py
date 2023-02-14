# -*- coding: utf-8 -*-
import re
import os
import sys
import tqdm
sys.path.append("../lib/post_ocr_correction")
from pathlib import Path
from pytorch_beam_search import seq2seq
from post_ocr_correction import correction


folder = "../../data/ICDAR2019_POCR_competition_dataset/ICDAR2019_POCR_competition_training_18M_without_Finnish/EN/"

output_folder = Path("../../data/en")
files = sorted(os.listdir(folder))
print("len files: %d" % len(files))

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

print("Data processing")
data = list(p.imap_unordered(extract, tqdm(files), chunksize=128))


# train data and model
source = [list("abcdefghijkl"), list("mnopqrstwxyz")]
target = [list("abcdefghijk"), list("mnopqrstwxy")]
source_index = seq2seq.Index(source)
target_index = seq2seq.Index(target)
X = source_index.text2tensor(source)
Y = target_index.text2tensor(target)
model = seq2seq.Transformer(source_index, target_index)
model.train()
model.fit(X, Y, epochs=100, progress_bar=0)
model.eval()

# test data
test = "ghijklmnopqrst"
new_source = [list(test)]
X_new = source_index.text2tensor(new_source)

# plain beam search
predictions, log_probabilities = seq2seq.beam_search(
    model,
    X_new,
    progress_bar=0
)
just_beam = target_index.tensor2text(predictions[:, 0, :])[0]
just_beam = re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", just_beam)

# post ocr correction
disjoint_beam = correction.disjoint(
    test,
    model,
    source_index,
    target_index,
    5,
    "beam_search",
)
_, n_grams_beam = correction.n_grams(
    test,
    model,
    source_index,
    target_index,
    5,
    "beam_search",
    "triangle"
)

print("\nresults")
print("  test data                      ", test)
print("  plain beam search              ", just_beam)
print("  disjoint windows, beam search  ", disjoint_beam)
print("  n-grams, beam search, triangle ", n_grams_beam)
