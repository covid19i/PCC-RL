#! /bin/bash
source /home/ubuntu/environments/my_env/bin/activate

RUN_NAME="LSTM_run2_1600x410_2048_1_lstm_dim_64"
echo $RUN_NAME
echo $$
echo "File Name: $0"

python3 stable_solve_LSTM.py --model-dir=/home/ubuntu/models/$RUN_NAME/ > ./$RUN_NAME.txt

mkdir $RUN_NAME
mv ./$RUN_NAME.txt $RUN_NAME
mv pcc_model_* pcc_env_log_run_.json $RUN_NAME
