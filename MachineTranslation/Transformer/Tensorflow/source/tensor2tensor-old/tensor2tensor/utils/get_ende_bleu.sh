#!/bin/bash

mosesdecoder=~/mosesdecoder

decodes_file=$1
gold_targets=$2

# Tokenize.
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decodes_file > $decodes_file.tok
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $gold_targets > $gold_targets.tok

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $gold_targets.tok > $gold_targets.tok.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.tok > $decodes_file.tok.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $gold_targets.tok.atat < $decodes_file.tok.atat
