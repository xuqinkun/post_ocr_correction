# -*- coding: utf-8 -*-
import torch
import torch.utils.data as tud
import torch.nn as nn
import pickle
from pathlib import Path
from tqdm.notebook import tqdm
import datetime
import matplotlib.pyplot as plt
import importlib
import sys
import pandas as pd

# from pytorch_decoding.seq2seq import Transformer
# from torch.nn import Transformer
from pytorch_beam_search.seq2seq import Transformer

scratch = Path("../../data/en/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
char2i = pickle.load(open(scratch / "data/char2i.pkl", "rb"))
i2char = pickle.load(open(scratch / "data/i2char.pkl", "rb"))

# Data
train_size = 1000000

dev_size = 1000000

train_source = torch.load(scratch / "data/train_source.pt")[:train_size].to(device)
print(train_source.shape)
train_target = torch.load(scratch / "data/train_target.pt")[:train_size].to(device)
print(train_target.shape)

dev_source = torch.load(scratch / "data/dev_source.pt")[:dev_size].to(device)
print(dev_source.shape)
dev_target = torch.load(scratch / "data/dev_target.pt")[:dev_size].to(device)
print(dev_target.shape)

# Model
model = Transformer(char2i,
                    i2char,
                    max_sequence_length=110,
                    embedding_dimension=512,
                    feedforward_dimension=2048,
                    attention_heads=8,
                    encoder_layers=4,
                    decoder_layers=4)
model.to(device)
log = model.fit(train_source,
                train_target,
                dev_source,
                dev_target,
                epochs=1,
                progress_bar=2,
                learning_rate=10 ** -4)


def predict(X,
            batch_size=128,
            progress_bar=False):
    """
    Evaluates the model on a dataset.

    Parameters
    ----------
    X: LongTensor of shape (examples, input_length)
        The input sequences of the dataset.

    Y: LongTensor of shape (examples, output_length)
        The output sequences of the dataset.

    criterion: PyTorch module
        The loss function to evalue the model on the dataset, has to be able to compare self.forward(X, Y) and Y
        to produce a real number.

    batch_size: int
        The batch size of the evaluation loop.

    progress_bar: bool
        Shows a tqdm progress bar, useful for tracking progress with large tensors.

    Returns
    -------
    loss: float
        The average of criterion across the whole dataset.

    error_rate: float
        The step-by-step accuracy of the model across the whole dataset. Useful as a sanity check, as it should
        go to zero as the loss goes to zero.

    """
    Y = torch.zeros_like(dev_source)
    Y[:, 0] = char2i['<START>']
    dataset = tud.TensorDataset(X, Y)
    loader = tud.DataLoader(dataset, batch_size=batch_size)
    model.eval()
    results = []
    probs = []
    with torch.no_grad():
        iterator = iter(loader)
        if progress_bar:
            iterator = tqdm(iterator)
        for batch in iterator:
            x, y = batch
            # compute loss
            p = model.forward(x, y).transpose(1, 2)[:, :, :-1]
            # compute accuracy
            predictions = p.argmax(1)
            results.append(predictions)
            probs.append(p)
    return results, probs


idx, probs = predict(dev_source)


def tensor2text(inputs):
    inputs = [ids.to("cpu") for ids in inputs]
    for doc in inputs:
        sents = doc.numpy().tolist()
        text = [[i2char[i] for i in ids] for ids in sents]




tensor2text(idx)
# model.tensor2text(idx)
# model.tensor2text(train_target[:10])
