COMMAND="python tools/train_net.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml  --box_min_ap 0.377 --mask_min_ap 0.339  --seed 3 "

if [ $# -eq 0 ]
then
        echo "nvprof disabled"

elif [ "$1" == "--profile" ]
then
        echo "nvprof is profiling compute utilization"
        COMMAND="nvprof --profile-from-start off --export-profile nvprof_data/compute_utilization%p.nvvp --print-summary ${COMMAND} --profile"

elif [ "$1" == "--profile-fp32" ]
then
        echo "nvprof is profiling fp32 utilization"
        COMMAND="nvprof --profile-from-start off --metrics single_precision_fu_utilization --export-profile nvprof_data/fp32_utilization%p.nvvp --print-summary ${COMMAND} --profile"

else
        echo "Invalid input argument. Valid ones are --profile --profile-fp32"
        return -1
fi

$COMMAND | tee run.log
