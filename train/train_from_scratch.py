# -*- coding: utf-8 -*-
import argparse
import json
import re
import sys

import torch

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
    train_data = list(
        pool.imap_unordered(create_windows, tqdm(zip(docs, [window_size for i in docs])), chunksize=128))
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
    pool = Pool(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-w', '--window_size', type=int)
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--do_prediction', action="store_true")
    args = parser.parse_args()

    folder = Path(args.input_path)
    model_path = Path(args.model_path)
    output_folder = Path(args.output_path)

    train_dir = folder / 'train'
    test_dir = folder / 'test'
    train_align_document = train_dir / 'align'
    test_align_document = train_dir / 'align'
    train_files = train_align_document.glob('*.txt')
    test_files = test_align_document.glob('*.txt')
    model_checkpoint = model_path / 'model.pt'

    window_size = args.window_size
    train_docs = list(pool.imap_unordered(extract, tqdm(train_files), chunksize=128))
    train_source_index, train_target_index, train_X, train_Y = load_data(train_docs)
    test_docs = list(pool.imap_unordered(extract, tqdm(test_files), chunksize=128))
    # load model
    if model_checkpoint.exists():
        model = torch.load(model_checkpoint)
    else:
        model = seq2seq.Transformer(train_source_index,
                                    train_target_index,
                                    max_sequence_length=train_X.shape[1]
                                    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    print(f'Device={device}')
    print(f'Window size={window_size}')
    if args.do_train:
        print('Train')
        model.train()
        X = train_X.to(device)
        Y = train_Y.to(device)
        model.fit(X_train=X, Y_train=Y, epochs=5, progress_bar=1)
        torch.save(model, model_checkpoint)
    if args.do_eval:
        print('Eval')
        model.eval()
        eval_source_index, eval_target_index, eval_X, eval_Y = load_data(test_docs)
        X_eval = eval_X.to(device)
        Y_eval = eval_Y.to(device)
        loss, error_rate = model.evaluate(X_eval, Y_eval, progress_bar=True)
        print(f'Loss={loss}\terror_rate={error_rate}')
    # do_prediction
    if args.do_prediction:
        print('Prediction')
        model.eval()
        model.to('cpu')
        test_data = list(
            pool.imap_unordered(create_windows, tqdm(zip(test_docs, [window_size for i in test_docs])),
                                chunksize=128))
        test_data = [d for doc in test_data for d in doc]
        # train data and model
        test_source = [t[0] for t in test_data][0]
        test_target = [t[1] for t in test_data][0]
        test_source = "".join(test_source).replace(GAP, '')
        test_source = list(test_source)
        X_test = train_source_index.text2tensor(test_source)
        # X_test = X_test.to(device)
        # plain beam search
        predictions, log_probabilities = seq2seq.beam_search(
            model,
            X_test,
            progress_bar=0
        )
        just_beam = train_target_index.tensor2text(predictions[:, 0, :])[0]
        just_beam = re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", just_beam)

        # post ocr correction
        decoding_method = "beam_search"
        weight_function = "uniform"
        disjoint_beam = correction.disjoint(
            test_source,
            model,
            train_source_index,
            train_target_index,
            window_size=window_size,
            decoding_method=decoding_method,
        )
        _, n_grams_beam = correction.n_grams(
            test_source,
            model,
            train_source_index,
            train_target_index,
            window_size=window_size,
            decoding_method=decoding_method,
            weighting=weight_function
        )

        print("\nresults")
        print(f"  test data\t\t{''.join(test_source)}")
        print(f"  test target\t\t{''.join(test_target)}")
        print("  plain beam search              ", just_beam)
        print("  disjoint windows, beam search  ", disjoint_beam)
        print(f"  n-grams, {decoding_method}, {weight_function}:{n_grams_beam} ", )
    pool.close()
    pool.terminate()
