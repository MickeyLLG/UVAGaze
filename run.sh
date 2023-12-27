#!/bin/bash

gpu=0


for ((pid=0;pid<=8;pid++))  # training and testing procedure using weights pre-trained on Gaze360.
do
  python3 adapt.py --i -1 --cams 18 --pic 256 --bs 32  --pairID ${pid} --savepath test/gaze3602eth --source gaze360 --target eth-mv --gpu ${gpu} --stb --pre
  python3 test_pair.py --pairID ${pid} --savepath test/gaze3602eth --target eth-mv100k --gpu ${gpu}
  python3 calc_metric.py --pairID ${pid} --savepath test/gaze3602eth --source gaze360 --target eth-mv100k
done

for ((pid=0;pid<=8;pid++))  # training and testing procedure using weights pre-trained on eth-mv.
do
  python3 adapt.py --i -1 --cams 18 --pic 256 --bs 32  --pairID ${pid} --savepath test/eth2eth --source eth-mv-train --target eth-mv --gpu ${gpu} --stb --pre
  python3 test_pair.py --pairID ${pid} --savepath test/eth2eth --target eth-mv100k --gpu ${gpu}
  python3 calc_metric.py --pairID ${pid} --savepath test/eth2eth --source eth-mv-train --target eth-mv100k
done
