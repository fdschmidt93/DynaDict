# DynaDict

## Description

DynaDict is a simple tool to induce an n-gram phrase table from a list of phrases and cross-lingual word embeddings of source and target language. Candidate phrases are first pre-ranking by cosine similarity of (averaged) phrase word embeddings and then re-ranked with [DynaMax-Jaccard](https://github.com/babylonhealth/fuzzymax).

## Walkthrough
1. Train word embeddings with [word2vec](https://github.com/tmikolov/word2vec) or [fastText](https://github.com/facebookresearch/fastText) for source and target language corpora
2. Map embeddings to a joint space with Procrustes or [vecmap](https://github.com/artetxem/vecmap)
3. Extract n-grams of source and target language corpora
4. Infer n-gram phrase table(s) with DynaDict
5. Optional: iteratively resolve multiple phrase tables

## Details

DynaDict segments step 4. of the **walkthrough** as follows:
1. **Pre-ranking:** Pre-rank top-5,000 candidates of the respective target language by phrase with cosine similarity of (averaged) phrase embeddings
2. **Re-ranking:** Re-rank top-5,000 candidates with fuzzy Jaccard similarity per DynaMax-Jaccard
3. **Candidate resolution:** Perform step 1 and 2 in both directions and add mutual nearest neighbors to dictionary

DynaDict is able to jointly infer any n-gram embeddings. In practice, we first inferred a joint dictionary for the top {50,100,100}K {uni,bi,tri}-grams, respectively, and, subsequently repeat the algorithm for all {uni,bi,tri}-grams stand-alone. The resulting four dictionaries are then resolved by sequentially adding phrase candidates to the joint dictionary from the n-gram dictionaries, for which none of the candidate phrases is yet included.

## Command-line Interface

```
n-gram Dictionary Induction using DynaMax Jaccard

positional arguments:
  PATH                  Path to mapped source embeddings, stored word2vec style
  PATH                  Path to mapped target embeddings, stored word2vec style
  PATH                  Path to source n-grams
  PATH                  Path to target n-grams
  PATH                  Path to store inferred dictionary
  N                     top-k candidates for to retrieve during pre-ranking

optional arguments:
  -h, --help            show this help message and exit
  --src_unigram_counts PATH
                        Path to source unigram counts for smooth inverse frequency weighting (default: None)
  --trg_unigram_counts PATH
                        Path to target unigram counts for smooth inverse frequency weighting (default: None)
```
* **Usage:** See `induce_dico.sh` and `merge_dico.sh` for guidance. `merge_dico.sh` enables to iteratively increment additional dictionaries, only adding candidates for which none of the phrases were included in the prior iteration(s).
* **Input format:** Phrases should be in a UTF-8 encoded file with one phrase per line and single tokens per phrase being whitespace-separated. If pre-ranking shall use [SIF](https://openreview.net/pdf?id=SyK00v5xx), then uni-grams and uni-gram probabilities should be tab-delimited (cf. `./samples/input/en.phrases.txt`) 

## Requirements

This code is written in Python 3. The requirements are listed in requirements.txt.

``pip3 install -r requirements.txt``

In particular, DynaDict requires NumPy and Numba to perform fast retrieval. To that end, DynaDict includes a JIT-compiled version of DynaMax-Jaccard to enable large-scale parallelization.

## FAQ

* **Why DynaMax-Jaccard?:** DynaMax-Jaccard is a high-performing non-parametric word embedding aggregator that performs strongly in cross-lingual retrieval scenarios, see [SEAGLE](https://www.aclweb.org/anthology/D19-3034.pdf) for a comparative evaluation
* **Why Pre-Ranking:?** DynaMax-Jaccard requires many expensive operations (dynamic vector creation, multiple pooling operations) that become prohibitively expensive with quadratic complexity
* **Why Numba?**: Numba is a JIT compiler that allows to straightforwardly parallelize non-vectorized operations
* **Why Iterative Candidate Resolution?**: Iterative resolution of dictionaries balances dictionary quality and size; mutual nearest neighbors are more regularized for larger candidate sets when inferring a joint dictionary

## References
 
* Vitalii Zhelezniak, Aleksandar Savkov, April Shen, Francesco Moramarco, Jack Flann, and Nils Y. Hammerla, [*Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors, ICLR 2019.*](https://openreview.net/forum?id=SkxXg2C5FX)
* Fabian David Schmidt, Markus Dietsche, Simone Ponzetto, Goran Glavas, [*SEAGLE: A Platform for Comparative Evaluation of Semantic Encoders for Information Retrieval, EMNLP 2019*](https://www.aclweb.org/anthology/D19-3034.pdf)
* Sanjeev Arora, Yingyu Liang, Tengyu Ma, [*A Simple But Tough-To-Beat Baseline for Sentence Embeddings, ICLR 2017*](https://openreview.net/pdf?id=SyK00v5xx)

## Contact

**Author:** Fabian David Schmidt\
**Affiliation:** University of Mannheim\
**E-Mail:** fabian.david.schmidt@hotmail.de
