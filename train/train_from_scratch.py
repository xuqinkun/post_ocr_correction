# -*- coding: utf-8 -*-
import json
import re
import os
import sys
import argparse
import json
import glob
import torch
import pickle

sys.path.append("../lib/post_ocr_correction")
from tqdm import tqdm
from pathlib import Path
from pytorch_beam_search import seq2seq
from post_ocr_correction import correction
from multiprocessing import Pool

GAP = '^'


def extract(name):
    with open(name) as file:
        text = file.read()
    doc = json.loads(text)
    return doc


def create_windows(x):
    doc, window_length = x
    gs_align = doc['gs_align']
    ocr_align = doc['ocr_align']
    assert len(gs_align) == len(ocr_align)
    return [(gs_align[i:i + window_length], ocr_align[i:i + window_length])
            for i in range(len(gs_align) + 1)]


def load_data(docs):
    source_save_file = train_dir / 'source.pt'
    target_save_file = train_dir / 'target.pt'
    source_index_file = train_dir / 'source_index.pt'
    target_index_file = train_dir / 'target_index.pt'
    if source_save_file.exists() and target_save_file.exists() and \
            source_index_file.exists() and target_index_file.exists():
        source_index = torch.load(source_index_file)
        target_index = torch.load(target_index_file)
        X = torch.load(source_save_file)
        Y = torch.load(target_save_file)
        return source_index, target_index, X, Y
    train_data = list(p.imap_unordered(create_windows, tqdm(zip(docs, [window_length for i in docs])), chunksize=128))
    train_data = [d for doc in train_data for d in doc]
    # train data and model
    source = ["".join(t[0]).replace(GAP, '') for t in train_data]
    source = [list(t) for t in source]
    target = [t[1] for t in train_data]
    source_index = seq2seq.Index(source)
    target_index = seq2seq.Index(target)
    X = source_index.text2tensor(source, progress_bar=True)
    Y = target_index.text2tensor(target, progress_bar=True)
    torch.save(source_index, source_index_file)
    torch.save(target_index, target_index_file)
    torch.save(X, source_save_file)
    torch.save(Y, target_save_file)
    return source_index, target_index, X, Y


if __name__ == '__main__':
    p = Pool(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = Path(args.input_path)
    model_path = Path(args.model_path)
    output_folder = Path(args.output_path)

    train_dir = folder / 'train'
    test_dir = folder / 'test'
    train_align_document = train_dir / 'align'
    test_align_document = train_dir / 'align'
    train_files = train_align_document.glob('*.txt')
    test_files = test_align_document.glob('*.txt')

    window_length = 100
    train_docs = list(p.imap_unordered(extract, tqdm(train_files), chunksize=128))
    test_docs = list(p.imap_unordered(extract, tqdm(test_files), chunksize=128))
    source_index, target_index, X, Y = load_data(train_docs)

    model = seq2seq.Transformer(source_index,
                                target_index,
                                max_sequence_length=X.shape[1]
                                )
    model.to(device=device)
    model.train()
    X = X.to(device)
    Y = Y.to(device)
    model.fit(X_train=X, Y_train=Y, epochs=100, progress_bar=1)
    model.eval()
    with open(model_path / 'model.pt', 'w') as f:
        torch.save(model, f)
    # test data

    test_data = list(
        p.imap_unordered(create_windows, tqdm(zip(test_docs, [window_length for i in test_docs])), chunksize=128))
    test_data = [d for doc in test_data for d in doc]
    # train data and model
    test_source = [t[0] for t in test_data][0:1]
    X_test = source_index.text2tensor(test_source)
    # plain beam search
    predictions, log_probabilities = seq2seq.beam_search(
        model,
        X_test,
        progress_bar=0
    )
    just_beam = target_index.tensor2text(predictions[:, 0, :])[0]
    just_beam = re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", just_beam)

    # post ocr correction
    disjoint_beam = correction.disjoint(
        test_source,
        model,
        source_index,
        target_index,
        5,
        "beam_search",
    )
    _, n_grams_beam = correction.n_grams(
        test_source,
        model,
        source_index,
        target_index,
        5,
        "beam_search",
        "triangle"
    )

    print("\nresults")
    print("  test data                      ", test_source)
    print("  plain beam search              ", just_beam)
    print("  disjoint windows, beam search  ", disjoint_beam)
    print("  n-grams, beam search, triangle ", n_grams_beam)
