import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
def get_parser():
    parser = argparse.ArgumentParser(
            description='n-gram Dictionary Induction using DynaMax Jaccard',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--supervision', metavar='PATH', type=str,
                        help='Path to tab-delimited unigram dictionary')
    parser.add_argument('--ngrams', metavar='PATH', nargs='+',
                        help='Path to tab-delimted uni-bi-tri-gram dictionaries')
    parser.add_argument('--output', metavar='PATH', type=str,
                        help='Path to output')
    return parser.parse_args()

def read_dico(path: str) -> dict:
    out = []
    with open(path, 'r') as file:
        for line in file:
            out.append(line.strip().split('\t'))
    return dict(out)

def refine(supervision: dict, dictionaries):
    """
    Join unigram and ngram dictionary:
    - if unigram and ngram disagree, take ngram
    - otherwise take unigram and ngram of set differences

    :param unigrams dict: src2trg unigram dictionary from mapping
    :param ngrams dict: post-hoc joint src2trg uni-bi-tri-gram induction
    """
    joint = supervision.copy()
    for dico in dictionaries:
        for k, v in dico.items():
            if k not in joint.keys():
                if v not in joint.values():
                    joint[k] = v
    return joint

def write_dico(dico: dict, path: str):
    with open(path, 'w') as file:
        for k, v in dico.items():
            file.write(f'{k}\t{v}\n')

def main():
    args = get_parser()
    supervision = read_dico(args.supervision)
    dictionaries = [read_dico(dico) for dico in args.ngrams]
    joint = refine(supervision, dictionaries)
    # joint = join(supervision, unigrams, ngrams)
    write_dico(joint, args.output)

if __name__ == '__main__':
    main()
