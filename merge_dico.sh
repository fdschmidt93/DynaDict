#!/usr/bin/bash

# SUPERVISION   reflects the base dictionary
# DICOS         list of paths to dictionaries which to iteratively resolve  
SUPERVISION="PATH/TO/PANLEX/DICO/TXT"
DICOS="PATH/TO/JOINT/DICO.txt PATH/TO/UNI-GRAM/DICO.txt PATH/TO/BI-GRAM/DICO.txt PATH/TO/TRI-GRAM/DICO.txt"
OUTPUT=/PATH/TO/OUTPUT/DICO/TXT
python merge.py --supervision $SUPERVISION --ngrams $DICOS --output $OUTPUT
