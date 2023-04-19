#!/bin/bash

# PYTHON_ENV=/home/wzielonka/miniconda3/etc/profile.d/conda.sh    # original
PYTHON_ENV=/home/bjgbiesseck/anaconda3/etc/profile.d/conda.sh     # Bernardo


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH

# export LD_LIBRARY_PATH=/is/software/nvidia/nccl-2.4.8-cuda10.1/lib/   # original
export LD_LIBRARY_PATH=/is/software/nvidia/nccl-2.4.8-cuda11.2/lib/     # Bernardo

source ${PYTHON_ENV}
# module load cuda/10.1  # original
# module load gcc/4.9    # original
# module load cuda/11.2    # Bernardo
# module load gcc/9        # Bernardo

EXPERIMENT=''                  # original
CHECKPOINT=''                  # original
BENCHMARK=''                   # original
PREDICTED=''                   # original
# EXPERIMENT='mica_duo.yml'    # Bernardo
# CHECKPOINT=''                # Bernardo
# BENCHMARK=''                 # Bernardo
# PREDICTED=''                 # Bernardo

echo 'Testing has started...'

if [ -n "$1" ]; then EXPERIMENT=${1}; fi
if [ -n "$2" ]; then CHECKPOINT=${2}; fi
if [ -n "$3" ]; then BENCHMARK=${3}; fi
if [ -n "$4" ]; then PREDICTED=${4}; fi

# ROOT=/home/wzielonka/projects/MICA/output/                                # original
# NOW=/home/wzielonka/datasets/NoWDataset/final_release_version/            # original
ROOT=/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/        # Bernardo
NOW=/datasets1/bjgbiesseck/NoWDataset/NoW_Dataset/final_release_version/    # Bernardo

# BERNARDO
image_set=val
# error_out_path=/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/testing/now/logs
error_out_path=/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/testing/now/logs/${EXPERIMENT}
method_identifier=${EXPERIMENT}
gt_mesh_folder=${NOW}/scans
gt_lmk_folder=${NOW}/scans_lmks_onlypp

# conda activate NFC   # original

# cd /home/wzielonka/projects/MICA                                                                                                      # original
cd /home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction                                                                              # Bernardo
# python test.py --cfg /home/wzielonka/projects/MICA/configs/${EXPERIMENT}.yml --test_dataset ${BENCHMARK} --checkpoint ${CHECKPOINT}   # original
# python test.py --cfg /home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/${EXPERIMENT}.yml --test_dataset ${BENCHMARK} --checkpoint ${CHECKPOINT}     # Bernardo
python test_multitask_facerecognition1.py --cfg /home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/${EXPERIMENT}.yml --test_dataset ${BENCHMARK} --checkpoint ${CHECKPOINT}   # Bernardo

# source /home/wzielonka/.virtualenvs/NoW/bin/activate   # original
# cd /home/wzielonka/projects/NoW                        # original
# python compute_error.py ${NOW} ${PREDICTED} true       # original
cd /home/bjgbiesseck/GitHub/BOVIFOCR_now_evaluation      # Bernardo
python compute_error.py ${NOW} ${PREDICTED} ${image_set} ${error_out_path} ${method_identifier} ${gt_mesh_folder} ${gt_lmk_folder}    # Bernardo

# Plot diagram
# source /home/wzielonka/.virtualenvs/NoW/bin/activate
# python cumulative_errors.py