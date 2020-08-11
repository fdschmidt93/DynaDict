import argparse
import logging

import numba as nb
import numpy as np

from src.dynamax import dynamax_jaccard
from src.utils import Vocabulary, dict2typed_dict, write_dico
from src.prerank import demean_embs, preranking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def get_parser():
    """Wrap arguments into function."""
    parser = argparse.ArgumentParser(
            description='n-gram Dictionary Induction using DynaMax Jaccard',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_emb', metavar='PATH', type=str,
                        help='Path to mapped source embeddings, stored word2vec style')
    parser.add_argument('trg_emb', metavar='PATH', type=str,
                        help='Path to mapped target embeddings, stored word2vec style')
    parser.add_argument('src_tokens', metavar='PATH', type=str,
                        help='Path to source n-grams')
    parser.add_argument('trg_tokens', metavar='PATH', type=str,
                        help='Path to target n-grams')
    parser.add_argument('output', metavar='PATH', type=str,
                        help='Path to store inferred dictionary')
    parser.add_argument('k', metavar='N', type=int, default=5000,
                        help='top-k candidates for to retrieve during pre-ranking')
    parser.add_argument('--src_unigram_counts', metavar='PATH', type=str,
                        help='Path to source unigram counts for smooth inverse frequency weighting')
    parser.add_argument('--trg_unigram_counts', metavar='PATH', type=str,
                        help='Path to target unigram counts for smooth inverse frequency weighting')
    return parser.parse_args()

@nb.njit(parallel=True)
def dynamax_loop(src: nb.typed.Dict,
                 trg: nb.typed.Dict,
                 src_emb: np.ndarray,
                 trg_emb: np.ndarray,
                 top_k: np.ndarray) -> np.ndarray:
    """
    Rerank top-k target candidates for source language with Dynamax-Jaccard.

    Pre-ranking via cosine similarity of averaged word embeddings.

    :param src nb.typed.Dict: id2pointer dict to get word embeddings
    :param trg nb.typed.Dict: id2pointer dict to get word embeddings
    :param src_emb np.ndarray: word embedding matrix for source language
    :param trg_emb np.ndarray: word embedding matrix for target language
    :param top_k np.ndarray: top-k preranked candidate indices for source language
    """
    N = len(src)
    M = len(trg)
    K = top_k.shape[1]
    argmax = np.empty(N, dtype=nb.int64)
    for i in nb.prange(N):
        scores = np.zeros(M, dtype=nb.float32)
        _src_emb = src_emb[src[i]]
        for j in range(K):
            idx = top_k[i, j]
            scores[idx] = dynamax_jaccard(_src_emb, trg_emb[trg[idx]])
        argmax[i] = scores.argmax()
    return argmax

@nb.njit(parallel=True)
def mutual_nn(src_argmax: np.ndarray, trg_argmax: np.ndarray) -> np.ndarray:
    """
    Infer mutual nearest neighbors.

    src_argmax and trg_argmax contain pointers to target/source nearest
    neighbors. Test whether nearest neighbor are mutual by equality check by
    cross-referencing.

    :param src_argmax np.ndarray: nearest target neighbors for source phrases
    :param trg_argmax np.ndarray: nearest source neighbors for target phrases
    """
    N = src_argmax.shape[0]
    M = trg_argmax.shape[0]
    src_argmax = np.stack((np.arange(N), src_argmax), axis=1)
    trg_argmax = np.stack((np.arange(M), trg_argmax), axis=1)
    mutual_neighbours = np.empty(N, dtype=np.bool_)
    for i in nb.prange(N):
        if i == trg_argmax[src_argmax[i, 1], 1]:
            mutual_neighbours[i] = True
        else:
            mutual_neighbours[i] = False
    return src_argmax[mutual_neighbours]

def dynamax_mnn(src: nb.typed.Dict, trg: nb.typed.Dict,
                src_emb: np.ndarray, trg_emb: np.ndarray,
                src_k: np.ndarray, trg_k: np.ndarray) -> np.ndarray:
    """
    Run Dynamax-Jaccard in both directions and infer mutual neighbors.

    :param src nb.typed.Dict: src_id2pointers dictionary
    :param trg nb.typed.Dict: trg_id2pointers dictionary
    :param src_emb np.ndarray: unnormalized word embeddings matrix for src lang
    :param trg_emb np.ndarray: unnormalized word embeddings matrix for trg lang
    :param src_k np.ndarray: preranked target candidates for source lanaguage
    :param trg_k np.ndarray: preranked source candidates for target lanaguage
    """
    logging.info('DynaMax: commencing first loop')
    src_argmax = dynamax_loop(src, trg, src_emb, trg_emb, src_k)
    logging.info('DynaMax: commencing second loop')
    trg_argmax = dynamax_loop(trg, src, trg_emb, src_emb, trg_k)
    logging.info('DynaMax: inferring mutual nearest neighbors')
    mnn = mutual_nn(src_argmax, trg_argmax)
    return mnn

def main():
    args = get_parser()

    # load data
    logging.info('Loading data..')
    src_vocab = Vocabulary.from_embeddings(args.src_emb, top_n_words=-1)
    trg_vocab = Vocabulary.from_embeddings(args.trg_emb, top_n_words=-1)

    # parse phrases to word2id pointers
    src_id2word, src_id2pointers = src_vocab.phrase_vocabulary(args.src_tokens)
    trg_id2word, trg_id2pointers = trg_vocab.phrase_vocabulary(args.trg_tokens)

    # smooth inverse frequency if available
    if args.src_unigram_counts and args.trg_unigram_counts:
        logging.info('Performing SIF weighting')
        src_vocab.sif_weighting(args.src_unigram_counts)
        trg_vocab.sif_weighting(args.trg_unigram_counts)
        logging.info('Embeddings weighted by SIF.')

    # convert pointers to nb.typed.dict
    src_id2pointers = dict2typed_dict(src_id2pointers)
    trg_id2pointers = dict2typed_dict(trg_id2pointers)
    logging.info('Data loaded! Now prefiltering..')

    # prefilter 
    src_mean = demean_embs(src_id2pointers, src_vocab.emb)
    trg_mean = demean_embs(trg_id2pointers, trg_vocab.emb)
    src_k = preranking(src_mean, trg_mean, args.k)
    trg_k = preranking(trg_mean, src_mean, args.k)
    logging.info('Filtering completed, now inferring dictionary with DynaMax-Jaccard')

    # scoring
    mnn = dynamax_mnn(src=src_id2pointers, trg=trg_id2pointers,
                      src_emb=src_vocab.emb, trg_emb=trg_vocab.emb,
                      src_k=src_k, trg_k=trg_k)
    logging.info(f'Process completed. Found {len(mnn)} matches. Writing to file.')
    write_dico(mnn, src_id2word, trg_id2word, args.output)

if __name__ == '__main__':
    main()
