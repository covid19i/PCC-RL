#! /bin/bash
source /home/ubuntu/environments/my_env/bin/activate
echo $$
echo "File Name: $0"
RUN_NAME="32x4layers_60loops_run2"
echo "RUN_NAME: $RUN_NAME"
python3 stable_solve_32x3_16x1.py --model-dir=/home/ubuntu/models/$RUN_NAME/ > ./$RUN_NAME.txt
#python3 stable_solve_32x3_16x1.py --model-dir=/home/ubuntu/models/$RUN_NAME/ > ./$RUN_NAME.txt

mkdir $RUN_NAME
mv ./$RUN_NAME.txt $RUN_NAME
mv pcc_* $RUN_NAME