import os
import pickle
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Optional, Union
from collections import defaultdict

import numba as nb
import numpy as np

INT_ARRAY = nb.int64[:]
FLOAT_ARRAY = nb.float32[:, :]

@dataclass
class Vocabulary:
    """
    Ancillary word2id, id2word, and embeddings container class.

    :attr len: int, number of tokens in Dictionary
    :attr word2id: dict, keys: tokens, values: id
    :attr id2word: dict, k-v pair inverse of word2id
    :attr embeddings: list[list[np.ndarray]], tokens embedded
                      -- N: number of words
                      -- d: word vector dimensionality
    """
    word2id: dict = field(default_factory=dict)
    id2word: dict = field(default_factory=dict)
    emb: np.ndarray = field(init=False)

    def __len__(self):
        return len(self.word2id)

    def __getitem__(self, index):
        return self.id2word[index]

    def add(self, word):
        """
        Add a new word to word2id & id2word.

        :param word:
            (a) str, word to be added, e.g. "hello"
            (b) list[str], words to be added, e.g. ["first", "second"]
        """
        if isinstance(word, list):
            for token in word:
                self.add(token)
        else:
            assert isinstance(word, str), "Passed argument not a string"
            if word not in self.word2id:
                len_ = len(self)
                self.word2id[word] = len_
                self.id2word[len_] = word

    def add_embeddings(self, embeddings: np.ndarray, dtype=np.float32):
        assert isinstance(embeddings, np.ndarray), 'Embeddings are no numpy arrays!'
        assert len(embeddings) == len(self.word2id), 'Length mismatch!'
        self.emb = np.asarray(embeddings, dtype=dtype)

    @classmethod
    def from_embeddings(cls,
                        path: str,
                        pass_header: bool = True,
                        top_n_words: int = 200_000,
                        normalize: bool = False,
                        dtype=np.float32):
        """
        Instantiate Dictionary from pretrained word embeddings.

        :param path: str, path to pretrained word embeddings
        :param top_n_words: int, restrict Dictionary top_n_words frequent words
        :return: Dictionary, populated word2id and id2word from document tokens
        """
        assert os.path.exists(path), f'{path} not found!'
        cls_ = cls()
        with open(path, 'r') as f:
            embeddings = []
            if pass_header:
                next(f)
            for idx, line in enumerate(f):
                if len(cls_) == top_n_words:
                    break
                token, vector = line.rstrip().split(' ', maxsplit=1)
                if token not in cls_.word2id:
                    embeddings.append(np.fromstring(vector.strip(), sep=' '))
                    cls_.add(token)
        cls_.emb = np.asarray(np.stack(embeddings), dtype=dtype)
        if normalize:
            norm = np.linalg.norm(cls_.emb, ord=2, axis=-1, keepdims=True)
            cls_.emb /= norm
        assert len(cls_.emb) == len(cls_.word2id), 'Reading error!'
        return cls_

    def phrase_vocabulary(self, path_to_phrases: str):
        phrases = []
        with open(path_to_phrases, 'r') as file:
            for line in file:
                tokens = line.strip().split('\t')[0]
                if tokens:
                    if ' ' in tokens:
                        tokens = tokens.split(' ')
                    phrases.append(tokens)
        # reorganize variables
        id2word = {}
        id2pointers = {}
        for phrase in phrases:
            # unigram
            if isinstance(phrase, str):
                try:
                    pointers = np.array(self.word2id[phrase])[None]
                except KeyError:
                    print(f'{phrase} not in word2id')
            elif isinstance(phrase, list):
                try:
                    pointers = np.array([self.word2id[tok] for tok in phrase])
                except KeyError:
                    print(f'{phrase} not in word2id')
                    continue
            # filter final edge cases starting with space, etc.
            id_ = len(id2word)
            # phrase = '&#32;'.join(phrase)
            if isinstance(phrase, list):
                phrase = ' '.join(phrase)
            id2word[id_] = phrase
            id2pointers[id_] = pointers
        return id2word, id2pointers

    @classmethod
    def from_dictionary(cls,
                        dict_: Dict[str, int],
                        embeddings: Optional[np.ndarray] = None):
        """Instantiate Vocabulary instance from word2id dictionary."""
        assert isinstance(dict_, dict), 'Please pass a dictionary!'
        cls_ = cls()
        cls_.word2id = dict_
        cls_.id2word = {v: k for k, v in dict_.items()}
        if embeddings is not None:
            assert len(cls_) == len(embeddings), 'Shapes do not align!'
            assert isinstance(embeddings, np.ndarray), 'Not an np.ndarray'
            cls_.emb = embeddings
        return cls_

    @classmethod
    def from_pretrained(cls,
                        word2id_path: str,
                        embeddings_path: str):
        cls_ = cls()
        with open(word2id_path, 'rb') as file:
            word2id = pickle.load(file)
            cls_.word2id = word2id

        cls_.id2word = {v: k for k, v in cls_.word2id.items()}
        cls_.emb = np.load(embeddings_path)
        return cls_

    def write(self, path, header=True):
        with open(path, 'w') as vec:
            if header:
                header_string = f'{len(self.word2id)} {self.emb.shape[-1]}'
                vec.write(header_string+'\n')
            for i, (word, emb) in enumerate(zip(self.word2id, self.emb)):
                if (i + 1) % 100_000 == 0:
                    print(f'{i+1} of {len(self.word2id)} phrases written to file!')
                out = word + ' ' + ' '.join(map(str, emb))
                vec.write(out+'\n')

    def sif_weighting(self, path: str, a: float=0.001):
        """
        Weight embedding matrices by smooth-inverse frequency.

        :param path str: path to token unigram-probability, tab-delimited
        :param a float: smoothing factor
        """
        with open(path, 'r') as file:
            tokens = []
            probabilities = []
            for line in file:
                token, word_probability = line.split('\t')
                tokens.append(token)
                probabilities.append(word_probability)
        probabilities = np.array(probabilities, dtype=np.float32)
        probabilities = a / (a + probabilities)
        for tok, prob in zip(tokens, probabilities):
            # scale embedding
            try:
                self.emb[self.word2id[tok]] *= prob
            except KeyError:
                continue

def dict2typed_dict(dico: dict):
    """
    Convert id2pointer Python to typed Numba Dict to benefit from JIT
    compilation speedups.

    :param dico dict: id2pointer (token id to list of word embedding pointers)
    """
    typed_dico = nb.typed.Dict.empty(key_type=nb.int64, value_type=INT_ARRAY)
    N = len(dico)
    for i in range(N):
        typed_dico[i] = dico[i]
    return typed_dico

def write_dico(mnn: np.ndarray, src_id2word: dict, trg_id2word: dict, path: str):
    """
    Write mutual nearest neighbors to tab-delimited UTF-8 encoded file.

    :param mnn np.ndarray: arr for which columns denote mutual neighbors
    :param src_id2word dict: map id to tokens for source language
    :param trg_id2word dict: map id to tokens for target language
    :param path str: where to write txt file
    """
    with open(path, 'w') as file:
        for i, j in mnn:
            file.write(f'{src_id2word[i]}\t{trg_id2word[j]}\n')
