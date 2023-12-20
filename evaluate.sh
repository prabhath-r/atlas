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
SIZE=large
DATA_DIR=/home/016032497/CMPE259/atlas/atlas_data
EVAL_FILE="${DATA_DIR}/test_10/test_exp10.jsonl"
SAVE_DIR=${DATA_DIR}/Linear_Experiments_Large
EXPERIMENT_NAME="Large_Exp10"

# submit your code to Slurm
srun python evaluate.py \
    --name 'Linear_Large_Eval10' \
    --generation_max_length 5 \
    --gold_score_mode "pdist" \
    --precision fp32 \
    --reader_model_type google/t5-${SIZE}-lm-adapt \
    --text_maxlength 128 \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}/checkpoint/step-1000 \
    --eval_data "${EVAL_FILE}" \
    --per_gpu_batch_size 1 \
    --n_context 20 --retriever_n_context 30 \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --index_mode flat  \
    --task multiple_choice \
    --load_index_path ${SAVE_DIR}/${EXPERIMENT_NAME}/saved_index \
    --write_results \
    --save_index_n_shards 1