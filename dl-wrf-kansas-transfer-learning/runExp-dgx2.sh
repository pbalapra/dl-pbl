#!/bin/bash
#COBALT -n 1
#COBALT -q dgx 
#COBALT -A Performance
#COBALT -t 6:00:00

export PATH=/soft/interpreters/python/intelpython/27/bin:$PATH

source activate dgx2-dl

for i in 55448 43775 29190 14601 1
do

python code/wrfmodel65.py --model_type=hpc --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64

python code/wrfmodel65.py --model_type=hac --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64

python code/wrfmodel65.py --model_type=mlp --start_id=$i --optimizer=adam --learning_rate=0.001 --epochs=1000 --batch_size=64


done

source deactivate








