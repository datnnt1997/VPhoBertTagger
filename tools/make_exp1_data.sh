#!/usr/bin/env bash

trainDir="../datasets/vlsp2018/raw/VLSP2018-NER-train-Jan14"
devDir="../datasets/vlsp2018/raw/VLSP2018-NER-dev"
testDir="../datasets/vlsp2018/raw/VLSP2018-NER-Test-Domains"

# For training data
dataDir="../datasets/vlsp2018/exp1/train"

mkdir -p $dataDir

python word_segment.py $trainDir $dataDir/train_syllables.txt $dataDir/train_ws.txt > $dataDir/log-train.txt

cut -f 1,2 -d " " $dataDir/train_ws.txt > $dataDir/train_ws-l1.txt
cut -f 1,3 -d " " $dataDir/train_ws.txt > $dataDir/train_ws-l2.txt
cut -f 1,4 -d " " $dataDir/train_ws.txt > $dataDir/train_ws-l3.txt
python join_tags2.py -o $dataDir/train_ws-l1+l2.txt $dataDir/train_ws-l1.txt $dataDir/train_ws-l2.txt

# For devset
dataDir="../datasets/vlsp2018/exp1/dev"

mkdir -p $dataDir

python word_segment.py $devDir $dataDir/dev_syllables.txt $dataDir/dev_ws.txt > $dataDir/log-dev.txt

cut -f 1,2 -d " " $dataDir/dev_ws.txt > $dataDir/dev_ws-l1.txt
cut -f 1,3 -d " " $dataDir/dev_ws.txt > $dataDir/dev_ws-l2.txt
cut -f 1,4 -d " " $dataDir/dev_ws.txt > $dataDir/dev_ws-l3.txt
python join_tags2.py -o $dataDir/dev_ws-l1+l2.txt $dataDir/dev_ws-l1.txt $dataDir/dev_ws-l2.txt

# For test set
dataDir="../datasets/vlsp2018/exp1/test"

mkdir -p $dataDir

python word_segment.py $devDir $dataDir/test_syllables.txt $dataDir/test_ws.txt > $dataDir/log-test.txt

cut -f 1,2 -d " " $dataDir/test_ws.txt > $dataDir/test_ws-l1.txt
cut -f 1,3 -d " " $dataDir/test_ws.txt > $dataDir/test_ws-l2.txt
cut -f 1,4 -d " " $dataDir/test_ws.txt > $dataDir/test_ws-l3.txt
python join_tags2.py -o $dataDir/test_ws-l1+l2.txt $dataDir/test_ws-l1.txt $dataDir/test_ws-l2.txt






