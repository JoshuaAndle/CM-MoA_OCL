#!/bin/bash

#SBATCH --job-name=OCL         # Job name
#SBATCH --cpus-per-task=1               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=200:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                    # Request one GPU


#SBATCH --output=./logs/baseline_Reverse%j.log   # Standard output and error log


echo "====================================================="
pwd; hostname; date
echo "====================================================="


echo "Running Adversarial CL Training"

module load apptainer


GPUS=0
NB_COMMA=`echo ${GPUS} | tr -cd , | wc -c`
NB_GPUS=$((${NB_COMMA} + 1))

NOTE="" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

METHOD="moa-clip"
PEFT_ENCODER='both' # both, text, image
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet

N_TASKS=4
N=100
M=0

GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

# VIS_CLASS="batch"
VIS_CLASS="all"

RAND_NM="--rnd_NM"
# RAND_NM=""

# ZS_TEST="--zero_shot_evaluation"
ZS_TEST=""

MEM_SIZE=5000

echo "Checking If Else"


if [ "$DATASET" == "cifar10" ]; then
    ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"

elif [ "$DATASET" == "cifar100" ]; then
    ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"


elif [ "$DATASET" == "tinyimagenet" ]; then
    ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"

elif [ "$DATASET" == "imagenet-r" ]; then
    ONLINE_ITER=3
    MODEL_NAME="ViT-B/16" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-4 OPT_NAME="adamw" SCHED_NAME="default"

else
    echo "Undefined setting"
    exit 1
fi









echo "Variables Set"


# for num_sets in 10 25 50 100 200 400
# do
for seed in 1
do
    note="${METHOD}_${VIS_CLASS}-${PEFT_ENCODER}_SEED${seed}"
    for task_num in 1
    # for task_num in 0
    do

        time srun apptainer run --nv ~/MoA_Container.sif \
        python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \
        --n_tasks $N_TASKS --m $M --n $N \
        --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
        --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
        --lr $LR --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
        --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
        --note $note --eval_period $EVAL_PERIOD --n_worker 4 --num_gpus ${NB_GPUS}




        # # time srun apptainer run --nv ~/pytorch-advcorr2.simg \
        # time srun apptainer run --nv ~/MoA_Container.sif --env INFO="${info}" --env CUDA_VISIBLE_DEVICES=${GPUS} \
        # python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \

        # time srun apptainer run --nv ~/MoA_Container.sif \
        # python "main.py" --method $METHOD --dataset $DATASET $ZS_TEST \
        # --n_tasks $N_TASKS --m $M --n $N \
        # --rnd_seed $seed --peft_encoder $PEFT_ENCODER \
        # --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
        # --lr $LR --batchsize $BATCHSIZE --visible_classes $VIS_CLASS \
        # --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
        # --note $note --eval_period $EVAL_PERIOD --n_worker 4 --num_gpus ${NB_GPUS} --debug --per_task_datasets




    done
done



echo "====================================================="
date





