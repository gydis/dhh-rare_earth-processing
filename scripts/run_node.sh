#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=40
#SBATCH --account=project_2014153
#SBATCH --job-name=test-run
#SBATCH --output=output/test_run_%j.out

# lang_sizes = {
#     "rus_Cyrl": 884688865,
#     "eng_Latn": 4388525961,
#     "fin_Latn": 34815601,
#     "zho_Hans": 1403640133,
#     "deu_Latn": 482053407,
#     "ben_Beng": 11043918,
#     "hin_Deva": 13651945
# }
export HF_HOME="/scratch/project_2014153/rare-earth/.cache/hf"
PROJECT_PATH="/scratch/project_2014153/rare-earth"

module load python-data
source "${PROJECT_PATH}/processing_env/bin/activate"

python "${PROJECT_PATH}/dhh-rare_earth-processing/data_process.py" \
    --lang rus_Cyrl \
    --output_dir "${PROJECT_PATH}/output/test_run" \
    --chunk_size 10000 \
    --chunk_ind 1 \
    --batch_size 1000 \
    --output_dir "${PROJECT_PATH}/processed_data/rus_Cyrl_test/" \