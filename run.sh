#!/bin/bash

gpu=0

for ((pid=0;pid<=8;pid++))
do
  python3 adapt.py --i -1 --cams 18 --pic 256 --bs 32  --pairID ${pid} --savepath test --source eth-mv-train --target eth-mv --gpu ${gpu} --stb --pre
  python3 test_pair.py --pairID ${pid} --savepath test --target eth-mv --gpu ${gpu}
done

