import numba as nb
import numpy as np

def demean_embs(id2pointer: nb.typed.Dict, embeddings: np.ndarray) -> np.ndarray:
    """
    Aggregate word embeddings by averaging on phrase-level.

    :param id2pointer nb.typed.dict: token id to word embedding pointers
    :param embeddings np.ndarray: word embedding matrix
    """
    N = len(id2pointer)
    embs = []
    for i in range(N):
        emb = embeddings[id2pointer[i]]
        if emb.shape[0] > 1:
            emb = emb.mean(0, keepdims=True)
        embs.append(emb)
    embs = np.vstack(embs)
    embs /= np.linalg.norm(embs, axis=1, ord=2, keepdims=True)
    return embs

@nb.njit(parallel=True)
def preranking(src: np.ndarray, trg: np.ndarray, k: int) -> np.ndarray:
    """
    Prerank using cosine similarity of (averaged) word embeddings.

    L2-normlization already performed vectorized in 'demean_embs'.

    :param src np.ndarray: demeaned source language phrase embeddings
    :param trg np.ndarray: demeaned target language phrase embeddings
    :param k int: top-k candidates to retain
    """
    N = src.shape[0]
    argsort = np.empty((N, k), dtype=nb.int64)
    for i in nb.prange(N):
        argsort[i] = (src[i] @ trg.T).argsort()[-k:]
    return argsort
