#!/bin/bash

help()
{
    echo "Usage: run [ -c | --cuda_device ] (default : 0)
           [ -m | --model_path ] (default : ./pretrained_model)
           [ -i | --input_path ] (required)
           [ -o | --output_path ] (required)
           [ -b | --batch_size ] (default : 4)
           [ -g | --gradient_accumulation ] (default : 1)
           [ -e | --epoch ] (default : 10.0)
           [ -h | --help ]"
    exit 2
}

SHORT=m:,i:,o:,b:,g:,e:,c:,h
LONG=model_path:,input_path:,output_path:,batch_size:,gradient_accumulation:,epoch:,cuda_device:,help
OPTS=$(getopt -a -n run --options $SHORT --longoptions $LONG -- "$@")

VALID_ARGUMENTS=$#

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
    help
fi

eval set -- "$OPTS"

while :
do
    case "$1" in
        -m | --model_path)
            model_path="$2"
            shift 2
            ;;
        -i | --input_path)
            input_path="$2"
            shift 2
            ;;
        -o | --output_path)
            output_path="$2"
            shift 2
            ;;
        -b | --batch_size)
            batch_size="$2"
            shift 2
            ;;
        -g | --gradient_accumulation)
            gradient_accumulation="$2"
            shift 2
            ;;
        -e | --epoch)
            epoch="$2"
            shift 2
            ;;
        -c | --cuda_device)
            cuda_device="$2"
            shift 2
            ;;
        -h | --help)
            help
            ;;
        --)
            shift;
            break
            ;;
        *)
            echo "Unexpected option: $1"
            help
            ;;
    esac
done

if [ -z "$output_path" ]
then
    echo "Missing --output_path " >&2
    help
    exit 1
fi

export LOCAL_RANK=${cuda_device:-0}
export CUDA_VISIBLE_DEVICES=${cuda_device:-0}
export OMP_NUM_THREADS=12

python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 29550 \
    --use-env \
    ./finetuning_t5.py \
        --do_train --do_eval \
        --model_name_or_path ${model_path:-./MODEL/qa/base} \
        --save_strategy epoch \
        --save_total_limit 10 \
        --label_smoothing_factor 0.3 \
        --train_file $input_path/train.jsonl \
        --validation_file $input_path/val.jsonl \
        --test_file $input_path/test.jsonl \
        --output_dir $output_path \
        --per_device_train_batch_size ${batch_size:-4} \
        --gradient_accumulation_steps ${gradient_accumulation:-1} \
        --num_train_epochs ${epoch:-1.0} \
        --overwrite_output_dir \
        --fp16
