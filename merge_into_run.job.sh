#!/bin/bash
# needed variables:
# files_run
# run
# merged_file

source "/home/yves.renier/miniconda3/bin/activate"
conda activate lst-dev

dl1merge_dir=$(dirname $merged_file)
run_dir=$dl1merge_dir/run${run}
if [ ! -e "$run_dir" ]; then
    if [ $(echo $files_run | wc -w) -le 0 ]; then 
        echo "ERROR: no DL1 file found for $date run $run with model $model";
        continue
    fi
    mkdir -p $run_dir
    for f in $files_run; do
        link_name="$run_dir/$(basename $f)"
        if [ ! -e "$link_name" ]; then
            ln -s $f $run_dir
        fi
    done
fi
#lst chain command to merge all file from a directory (removing images, keeping only Hillas parameters)
lstchain_merge_hdf5_files -d $run_dir -o $merged_file --smart False --no-image True &> ${merged_file/.h5/.log}

