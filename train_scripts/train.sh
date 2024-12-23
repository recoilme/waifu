#/bin/bash
set -e

work_dir=output/debug
np=4


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="/workspace/waifu/configs/sana_config/576ms/waifu-2b-576.yaml"
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=15432 \
        train_scripts/train_waifu.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=33 \
        --load_from=/workspace/waifu-2b-v01.pth \
        --debug=false \
        "$@"