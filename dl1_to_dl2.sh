#!/bin/bash

#list of runs for each day: http://www.lst1.iac.es/wiki/index.php/Daily_summary
dates="20200228 20200227 20200218 20200217 20200215 20200213 20200202 20200201 20200131 20200128 20200127 20200118 20200117 20200115 20200114 20191129 20191126 20191124 20191123"
models="jakub"

for date in $dates; do
    dl1merge_dir="$PWD/real_data/DL1/$date"
    #list merged files by runs
    files=$(ls $dl1merge_dir/*.h5)
    #get list of unique runs
    runs=$(echo $files | tr ' ' '\n' | sed 's|.*[rR]un\([0-9]\+\).*|\1|' | sort| uniq)
    for run in $runs; do
        export merged_file="$dl1merge_dir/run${run}_merged.h5"
        for model in $models; do
            rf_path="$PWD/models/$model"
            export rf_energy="$(ls $rf_path/energy_rf/*.joblib)"
            export rf_disp="$(ls $rf_path/disp_rf/*.joblib)"
            export rf_gh="$(ls $rf_path/gh_sep_rf/*.joblib)"
            dl2_dir="$PWD/real_data/DL2/$date/$model"
            mkdir -p $dl2_dir
            sleep 0.5
            export dl2file="$dl2_dir/run${run}.h5"
            if [ -e $dl2file ]; then
                echo "file $dl2file exist, skipping its creation"
                continue
            fi
            echo "converting run $run into DL2 using $model model:"
            job_log=${dl2file/.h5/}.job
            #use 16 CPUs per job
            sbatch -J "dl2_${run}" -o ${job_log}.out -e ${job_log}.err -N 16 -n 16 -p compute --mem=8000 -t 2:00:00 dl1_to_dl2.job.sh
            sleep 0.1
        done
    done
done
