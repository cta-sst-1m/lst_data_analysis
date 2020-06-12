#!/bin/bash

# needed variables:
# rf_energy
# rf_disp
# rf_gh
# dl2file
# merged_file

source "/home/yves.renier/miniconda3/bin/activate"
conda activate lst-dev

log_file="${dl2file/.h5/}.log"
#use jakub's script to create a DL2 file from a DL1 file for given RFs
./simulations/scripts/all_reco.py --energy_model=$rf_energy --disp_model=$rf_disp --sep_model=$rf_gh --input=$merged_file --output=$dl2file &> $log_file
