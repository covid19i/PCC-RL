#! /bin/bash
source /home/ubuntu/environments/my_env/bin/activate
echo $$
echo "File Name: $0"
RUN_NAME="LSTM_run$1_1600x410_loops_$2_timesteps_$3_1_lstm_dim_$4_shared_$5"
echo "RUN_NAME: $RUN_NAME"
python3 stable_solve_LSTM.py --model-dir=/home/ubuntu/models/$RUN_NAME/ --shared_arch=$5 --lstm_dim=$4 --no_of_timesteps=$3 --no_of_training_loops=$2 --RUN_ID=LSTM_run$1 > ./$RUN_NAME.txt

mkdir $RUN_NAME
mv ./$RUN_NAME.txt $RUN_NAME
mv pcc_* $RUN_NAME