#!/usr/bin/env bash

trainDir="../datasets/vlsp2018/raw/VLSP2018-NER-train-Jan14"
devDir="../datasets/vlsp2018/raw/VLSP2018-NER-dev"
testDir="../datasets/vlsp2018/raw/VLSP2018-NER-Test-Domains"
tmpDir="./work_dir"
dataDir="../datasets/vlsp2018/"

mkdir -p $dataDir
# Train data
python word_segment.py $trainDir $dataDir/train_syllables.txt $dataDir/train.txt --tmpdir $tmpDir --sent_tokenize
# Evaluate data
python word_segment.py  $devDir $dataDir/dev_syllables.txt $dataDir/dev.txt --tmpdir $tmpDir --sent_tokenize
# Testi data
python word_segment.py $testDir $dataDir/test_syllables.txt $dataDir/test.txt --tmpdir $tmpDir --sent_tokenize

rm -r $tmpDir