import numpy as np
import numba as nb

# Reimplementing DynaMax Jaccard in Numba to enable parallelilization and single
# execution speed ups of factor 6

@nb.njit
def np_max(arr, axis):
    """
    Workaround for np.max along axis in Numba.

    Credits to: https://github.com/numba/numba/issues/1269#issuecomment-472574352

    :param arr np.ndarray: stacked word embeddings
    :param axis int: axis for whichto get maximum
    """
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = np.max(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = np.max(arr[i, :])
    return result

@nb.njit
def fuzzify(s, u):
    """
    Sentence fuzzifier.
    Computes membership vector for the sentence S with respect to the
    universe U
    :param s: list of word embeddings for the sentence
    :param u: the universe matrix U with shape (K, d)
    :return: membership vectors for the sentence
    """
    f_s = s @ u.T
    m_s = np_max(f_s, axis=0)
    m_s = np.maximum(m_s, 0, m_s)
    return m_s

@nb.njit(fastmath=True)
def dynamax_jaccard(x, y):
    """
    DynaMax-Jaccard similarity measure between two sentences

    Credits to:
    -- Title: Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors
    -- Authors: Vitalii Zhelezniak, Aleksandar Savkov, April Shen, Francesco Moramarco, Jack Flann, Nils Y. Hammerla
    -- Published: ICLR 2019
    -- Paper: https://arxiv.org/pdf/1904.13264.pdf
    -- Github: https://github.com/babylonhealth/fuzzymax

    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    # feature generation
    u = np.vstack((x, y))
    m_x = fuzzify(x, u)
    m_y = fuzzify(y, u)
    # fuzzy jaccard
    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union
