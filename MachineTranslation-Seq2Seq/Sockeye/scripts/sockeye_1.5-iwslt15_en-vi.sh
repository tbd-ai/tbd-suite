#!/bin/sh

SOCKEYE_ROOT=$(cd $(dirname $0) && pwd)/..

mkdir -p $SOCKEYE_ROOT/measurements
rm -f $SOCKEYE_ROOT/measurements/$(basename $0).nvvp

PREFIX=
MAX_UPDATES=

if [ "$1" = "" ]
then
	PREFIX=
elif [ "$1" = "--profile" ]
then
	PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
		--export-profile $SOCKEYE_ROOT/measurements/$(basename $0).nvvp --print-summary"
	MAX_UPDATES="--max-updates 520"
elif [ "$1" = "--profile-fp32" ]
then
	PREFIX="/usr/local/cuda/bin/nvprof --profile-from-start off \
		--export-profile $SOCKEYE_ROOT/measurements/$(basename $0).nvvp --print-summary \
		--metrics single_precision_fu_utilization"
	MAX_UPDATES="--max-updates 520"
else 
	echo "Invalid input argument. Valid ones are --profile/--profile-fp32."; exit -1
fi

cd $SOCKEYE_ROOT/source/sockeye-1.5 && \
$PREFIX python3 -m sockeye.train --source $SOCKEYE_ROOT/dataset/train-preproc.en \
                                 --target $SOCKEYE_ROOT/dataset/train-preproc.vi \
                                 --validation-source $SOCKEYE_ROOT/dataset/tst2012.en \
                                 --validation-target $SOCKEYE_ROOT/dataset/tst2012.vi \
                                 --source-vocab $SOCKEYE_ROOT/dataset/vocab.en \
                                 --target-vocab $SOCKEYE_ROOT/dataset/vocab.vi \
                                 --output $SOCKEYE_ROOT/models/$(basename $0) --overwrite-output \
                                 --encoder rnn --decoder rnn \
                                 --num-layers 2:2 \
                                 --rnn-cell-type lstm \
                                 --rnn-num-hidden 512 \
                                 --rnn-encoder-reverse-input \
                                 --num-embed 512:512 \
                                 --attention-type mlp --attention-num-hidden 512 \
                                 --batch-size 32 \
                                 --bucket-width 10 \
                                 --metrics perplexity \
                                 --optimized-metric bleu \
                                 --checkpoint-frequency 1000 \
                                 --max-num-checkpoint-not-improved 80 \
                                 --weight-init uniform --weight-init-scale 0.1 \
                                 --learning-rate-reduce-factor 1.0 \
                                 --monitor-bleu -1 $MAX_UPDATES

