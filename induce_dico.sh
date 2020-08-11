#!/usr/bin/bash

# Illustrative input configuration linked to ./samples/{en,si}.phrases.txt
SRC='en'
TRG='si'
EMB_SRC=../PATH/TO/SOURCE-LANG/EMBEDDINGS.VEC
EMB_TRG=../PATH/TO/TARGET-LANG/EMBEDDINGS.VEC
SRC_PHRASES=./samples/input/${SRC}.phrases.txt
TRG_PHRASES=./samples/input/${SRC}.phrases.txt
OUTPUT=./samples/output/${SRC}-${TRG}.dynadict.txt
k=5000
python main.py $EMB_SRC $EMB_TRG $SRC_PHRASES $TRG_PHRASES $OUTPUT $k
