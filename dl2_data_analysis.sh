#!/bin/bash

#conda activate lst-dev

dates="20200218 20200217 20200215 20200202 20200201 20200131 20200128 20200127 20200118 20200117 20200115 20191129 20191126 20191124 20191123"

models="jakub"
analysis_version="v0.4.4_v00"

for date in $dates; do
    for model in $models; do
        python ./dl2_data_analysis.py --date $date --model=$model
    done
done
