# -*- coding: utf-8 -*-
import argparse
import json
import re
import sys
import random
import torch

POCR = 'POCR'

SROIE = 'SROIE'

sys.path.append("../lib/post_ocr_correction")
from tqdm import tqdm
from pathlib import Path
from pytorch_beam_search import seq2seq
from post_ocr_correction import correction
from multiprocessing import Pool

GAP = '^'


def _remove_bracket(text: str):
    idx = text.find(']')
    return text[idx + 1:]


def extract(args):
    name, dataset = args
    if dataset == SROIE:
        with open(name) as file:
            text = file.read()
        doc = json.loads(text)
        return doc
    else:
        with open(name) as file:
            lines = file.readlines()
        ocr_input, ocr_align, gs_align = lines
        ocr_input = _remove_bracket(ocr_input).lstrip()
        ocr_align = _remove_bracket(ocr_align).lstrip()[1:]
        gs_align = _remove_bracket(gs_align).lstrip()
        return {
            'ocr_input': ocr_input,
            'ocr_align': ocr_align,
            'gs_align': gs_align,
        }


def create_windows(args):
    doc, (window_length, dataset) = args
    gs_align = doc['gs_align']
    ocr_align = doc['ocr_align']
    return [(gs_align[i:i + window_length], ocr_align[i:i + window_length])
            for i in range(len(gs_align) + 1)]


def load_data(save_dir, docs, dataset, overwrite=False):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    print(f'Load data from {save_dir}')
    source_save_file = save_dir / 'source.pt'
    target_save_file = save_dir / 'target.pt'
    source_index_file = save_dir / 'source_index.pt'
    target_index_file = save_dir / 'target_index.pt'
    if source_save_file.exists() and target_save_file.exists() and \
            source_index_file.exists() and target_index_file.exists() and not overwrite:
        source_index = torch.load(source_index_file)
        target_index = torch.load(target_index_file)
        X = torch.load(source_save_file)
        Y = torch.load(target_save_file)
        return source_index, target_index, X, Y
    train_data = list(
        pool.imap_unordered(create_windows, tqdm(zip(docs, [(window_size, dataset) for i in docs])), chunksize=128))
    train_data = [d for doc in train_data for d in doc]
    # train data and model
    if dataset == SROIE:
        source = ["".join(t[0]).replace(GAP, '') for t in train_data]
    else:
        source = ["".join(t[0]).replace('@', '') for t in train_data]
    target = ["".join(t[1]) for t in train_data]
    source_index = seq2seq.Index(source)
    target_index = seq2seq.Index(target)
    X = source_index.text2tensor(source, progress_bar=True)
    Y = target_index.text2tensor(target, progress_bar=True)
    torch.save(source_index, source_index_file)
    torch.save(target_index, target_index_file)
    torch.save(X, source_save_file)
    torch.save(Y, target_save_file)
    return source_index, target_index, X, Y


def get_files(folder, dataset):
    if dataset == SROIE:
        train_dir = folder / 'train'
        test_dir = folder / 'test'
        train_align_document = train_dir / 'align'
        test_align_document = test_dir / 'align'
        train_set = [f for f in train_align_document.glob('*.txt')]
        dev_set = [f for f in test_align_document.glob('*.txt')]
        return train_set, dev_set
    else:
        train_dir = folder / 'ICDAR2019_POCR_competition_training_18M_without_Finnish'
        dev_dir = folder / 'ICDAR2019_POCR_competition_evaluation_4M_without_Finnish'
        en_train_dir = train_dir / 'EN' / 'EN1'
        en_dev_dir = dev_dir / 'EN' / 'EN1'
        train_set = [f for f in en_train_dir.glob('*.txt')]
        dev_set = [f for f in en_dev_dir.glob('*.txt')]
        return train_set, dev_set


if __name__ == '__main__':
    pool = Pool(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-ds', '--dataset', type=str, choices=[SROIE, POCR])
    parser.add_argument('-w', '--window_size', type=int, default=20)
    parser.add_argument('-e', '--epoch', type=int, default=5)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-do', '--dropout', type=float, default=0.5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001)
    parser.add_argument('-pb', '--progress_bar', type=int, default=0)
    parser.add_argument('-el', '--encoder_layers', type=int, default=2)
    parser.add_argument('-dl', '--decoder_layers', type=int, default=2)
    parser.add_argument('-ed', '--embedding_dimension', type=int, default=256)
    parser.add_argument('-fd', '--feedforward_dimension', type=int, default=1024)
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--do_prediction', action="store_true")
    parser.add_argument('--overwrite_model', action="store_true", default=False)
    parser.add_argument('--overwrite_eval', action="store_true", default=False)
    parser.add_argument('--overwrite_train', action="store_true", default=False)
    args = parser.parse_args()

    data_folder = Path(args.input_path)
    model_path = Path(args.model_path)
    output_folder = Path(args.output_path)
    epoch = args.epoch
    dataset = args.dataset
    window_size = args.window_size

    train_files, dev_files = get_files(data_folder, dataset)
    model_checkpoint = model_path / f'model_{dataset}_{epoch}_{window_size}.pt'

    train_docs = list(pool.imap_unordered(extract, zip(train_files, [dataset for f in train_files])))
    dev_docs = list(pool.imap_unordered(extract, zip(dev_files, [dataset for i in dev_files])))

    train_source_index, train_target_index, train_X, train_Y = load_data(save_dir=data_folder / 'train',
                                                                         docs=train_docs,
                                                                         dataset=dataset,
                                                                         overwrite=args.overwrite_train,
                                                                         )
    # load model
    if model_checkpoint.exists() and not args.overwrite_model:
        print(f'Load model from checkpoint {model_checkpoint}')
        model = torch.load(model_checkpoint)
    else:
        model = seq2seq.Transformer(train_source_index,
                                    train_target_index,
                                    max_sequence_length=train_X.shape[1],
                                    dropout=args.dropout,
                                    embedding_dimension=args.embedding_dimension,
                                    feedforward_dimension=args.feedforward_dimension,
                                    encoder_layers=args.encoder_layers,
                                    decoder_layers=args.decoder_layers,
                                    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    print(f'Device={device}')
    print(f'Window size={window_size}')
    # Train
    batch_size = args.batch_size
    if args.do_train and args.overwrite_model:
        print(f'Train on {dataset}')
        model.train()
        X = train_X.to(device)
        Y = train_Y.to(device)

        model.fit(X_train=X, Y_train=Y,
                  epochs=epoch,
                  batch_size=batch_size,
                  learning_rate=args.learning_rate,
                  progress_bar=args.progress_bar,
                  weight_decay=args.weight_decay,
                  )
        torch.save(model, model_checkpoint)
    if args.do_eval:
        print('Eval')
        model.eval()
        eval_source_index, eval_target_index, eval_X, eval_Y = load_data(save_dir=data_folder / 'test',
                                                                         docs=dev_docs,
                                                                         dataset=dataset,
                                                                         overwrite=args.overwrite_eval,
                                                                         )
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
            pool.imap_unordered(create_windows, tqdm(zip(dev_docs, [(window_size, dataset) for i in dev_docs])),
                                chunksize=128))
        for doc in test_data:
            lines = [d for d in doc]
            source = [line[0] for line in lines]
            target = [line[1] for line in lines]
            # train data and model
            if dataset == SROIE:
                ocr_input = [t.replace(GAP, '') for t in source]
            else:
                ocr_input = [t.replace('@', '') for t in source]
            full_text = "".join([s[0] for s in ocr_input if s != ''])
            gs_align = "".join([t[0] for t in target])
            X_test = train_source_index.text2tensor(ocr_input[0:1], progress_bar=True)
            X_test = X_test.to(device)
            model.to(device)
            # plain beam search
            metrics = correction.full_evaluation(
                raw=[full_text],
                gs=[gs_align],
                model=model,
                source_index=train_source_index,
                target_index=train_target_index,
                device=device,
            )
            print(metrics)
            # predictions, log_probabilities = seq2seq.beam_search(
            #     model,
            #     X_test,
            #     progress_bar=1,
            #     batch_size=batch_size,
            # )
            # just_beam = train_target_index.tensor2text(predictions[:, 0, :])
            # just_beam = re.sub(r"<START>|<PAD>|<UNK>|<END>.*", "", just_beam)
            #
            # # post ocr correction
            # decoding_method = "beam_search"
            # weight_function = "uniform"
            # disjoint_beam = correction.disjoint(
            #     ocr_input,
            #     model,
            #     train_source_index,
            #     train_target_index,
            #     window_size=window_size,
            #     decoding_method=decoding_method,
            #     document_batch_progress_bar=1,
            #     device=device,
            # )
            # _, n_grams_beam = correction.n_grams(
            #     ocr_input,
            #     model,
            #     train_source_index,
            #     train_target_index,
            #     window_size=window_size,
            #     decoding_method=decoding_method,
            #     weighting=weight_function,
            #     document_batch_progress_bar=1,
            #     device=device,
            # )
            # print("\n+++++++++Results+++++++++")
            # print(f"OCR input:\t\t{''.join([t[0] for t in ocr_input])}")
            # print(f"GS align\t\t{gs_align}")
            # print(f"Plain beam search {just_beam}")
            # print(f"Disjoint windows, beam search  ", disjoint_beam)
            # print(f"n-grams, {decoding_method}, {weight_function}:{n_grams_beam} ", )
    pool.close()
    pool.terminate()
