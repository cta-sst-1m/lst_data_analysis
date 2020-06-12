#!/bin/bash

dates="20200228 20200227 20200218 20200217 20200215 20200202 20200201 20200131 20200128 20200127 20200118 20200117 20200115 20191129 20191126 20191124 20191123"

analysis_version="v0.4.4_v00"

for date in $dates; do
    dl1_data_dir="/fefs/aswg/data/real/DL1/$date/$analysis_version"
    dl1merge_dir="$PWD/real_data/DL1/$date"
    files=$(ls $dl1_data_dir/*.h5)
    #look for run in the DL1 data directory
    runs=$(echo $files | tr ' ' '\n' | sed 's|.*[rR]un\([0-9]\+\).*|\1|' | sort| uniq)
    for run in $runs; do
        export files_run=$(echo $files | tr ' ' '\n' | grep -E "[rR]un${run}")
        export run=$run
        export merged_file="$dl1merge_dir/run${run}_merged.h5"
        mkdir -p $(dirname $merged_file)
        sleep 0.1
        if [ -e "$merged_file" ]; then
            echo "not merging run $run as $merged_file exists"
            continue
        fi
        echo "merging files for run $run on $date"
        job_log=${merged_file/.h5/}.job
        echo "log in $job_log"
        sbatch -o ${job_log}.out -e ${job_log}.err -N 1 -n 1 -p compute --mem=2000 -t 2:00:00 merge_into_run.job.sh
        sleep 0.1
    done
done
