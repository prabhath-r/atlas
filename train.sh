#!/bin/bash
#SBATCH --mail-user=prabhathreddy.gujavarthy@sjsu.edu
#SBATCH --mail-user=/dev/null
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=gpuTest_016032497
#SBATCH --output=gpuTest_%j.out
#SBATCH --error=gpuTest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00     
##SBATCH --mem-per-cpu=2000
##SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu   

source /opt/ohpc/pub/apps/anaconda/3.9/etc/profile.d/conda.sh 
conda activate my_env

export http_proxy=http://172.16.1.2:3128; export https_proxy=http://172.16.1.2:3128

cd /home/016032497/CMPE259/atlas

port=$(shuf -i 15000-16000 -n 1)
size=large
DATA_DIR=/home/016032497/CMPE259/atlas/atlas_data

TRAIN_FILE="${DATA_DIR}/train_10/train_exp1.jsonl"
EVAL_FILE="${DATA_DIR}/dev_10/dev_exp1.jsonl"
PASSAGE_FILE="${DATA_DIR}/nq_data/passage.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
SAVE_DIR=${DATA_DIR}/Linear_Experiments_Large/
EXPERIMENT_NAME=Large_Exp1
PRECISION="fp32" # or "fp32"
STEPS=1000

srun python train.py \
    --shuffle \
    --train_retriever --gold_score_mode ppmean \
    --use_gradient_checkpoint_reader \
    --use_gradient_checkpoint_retriever \
    --precision ${PRECISION} \
    --shard_optim --shard_grads \
    --temperature_gold 0.1 --temperature_score 0.1 \
    --refresh_index -1 \
    --target_maxlength 16 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --dropout 0.1 \
    --lr 1e-4 --lr_retriever 1e-5 \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 128 \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILE} \
    --per_gpu_batch_size 1 \
    --n_context 20 --retriever_n_context 30 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --total_steps ${STEPS} \
    --save_freq ${STEPS} \
    --main_port $port \
    --eval_freq 50   \
    --log_freq 10  \
    --write_results \
    --task multiple_choice \
    --multiple_choice_train_permutations all \
    --multiple_choice_eval_permutations cyclic\
    --passages ${PASSAGE_FILE} \
    --warmup_steps 10 \
    --save_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index \
    --index_mode flat \
    --query_side_retriever_training \
    --save_index_n_shards 1 \