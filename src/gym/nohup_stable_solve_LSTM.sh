#! /bin/bash
cd /home/ubuntu/duplicate-for-training/PCC-RL/src/gym
source /home/ubuntu/environments/my_env/bin/activate
nohup python3 stable_solve_LSTM.py --model-dir=/home/ubuntu/models/LSTM_nminibatch_1_n_steps_2048_loops_6_lstm_dim_256/ > ./LSTM_nminibatch_1_n_steps_2048_loops_6_lstm_dim_256.txt
