#!/bin/bash

DS2_ROOT=$(cd $(dirname $0) && pwd)/..

mkdir -p $DS2_ROOT/measurements
rm -f $DS2_ROOT/measurements/$(basename $0).nvvp

PREFIX=

case $1 in
	--profile)
		PREFIX="/usr/local/cuda/bin/nvprof \
			--profile-from-start off \
			--export-profile $DS2_ROOT/measurements/$(basename $0).nvvp \
			--print-summary"
		;;
	--profile-fp32)
		PREFIX="/usr/local/cuda/bin/nvprof \
			--profile-from-start off \
			--export-profile $DS2_ROOT/measurements/$(basename $0).nvvp \
			--print-summary \
			--metrics single_precision_fu_utilization"
		;;
	*)
		if [[ $# -gt 0 ]]; then
			echo "Usage: ./ds2-librispeech.sh [--profile|--profile-fp32]"
			exit -1
		fi
esac

pushd $DS2_ROOT/source
mkdir -p checkpoints
mkdir -p log
$PREFIX python main.py --configfile deepspeech.cfg
popd
