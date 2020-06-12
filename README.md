The sequence is:
1) merge_into_run.sh: regroup all pre-analysed DL1 subruns into a single file (also removing images so the file is small)
2) dl1_to_dl2.sh: apply the random forest to the DL1 data to produce DL2 data
3) dl2_data_analysis.sh apply cuts and produce plots (significance maps, rate of excess events, ...)

