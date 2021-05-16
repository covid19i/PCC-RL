# PCC-RL
Reinforcement learning resources for the Performance-oriented Congestion Control
project.

## Overview
This repo contains the gym environment required for training reinforcement
learning models used in the PCC project along with the Python module required to
run RL models in the PCC UDT codebase found at github.com/PCCProject/PCC-Uspace.


## Training
To run training only, go to ./src/gym/, install any missing requirements for
stable\_solve.py and run that script. By default, this should replicate the
model presented in A Reinforcement Learning Perspective on Internet Congestion
Control, ICML 2019.

## Testing Models

To test models in the real world (i.e., sending real packets into the Linux
kernel and out onto a real or emulated network), download and install the PCC
UDT code from github.com/PCCProject/PCC-Uspace. Follow the instructions in that
repo for using congestion control algorithms with Python modules, and see
./src/gym/online/README.md for additional instructions regarding testing or training models in the real world.


## Installation and sequence of instructions

For installation (on AWS Ubuntu 18.04 image):
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update && sudo apt-get install git cmake libopenmpi-dev python3.5-dev zlib1g-dev python3-pip python3.5-venv python3-testresources tmux sysstat

LSTMs trained without MPI libraries. Use the following instead for LSTMs.
(For using GPUs, don't install MPI: sudo apt-get update && sudo apt-get install git cmake  python3.5-dev zlib1g-dev python3-pip python3.5-venv python3-testresources tmux sysstat)

mkdir environments
cd environments
python3.5 -m venv my_env
source ~/environments/my_env/bin/activate

PCC-Uspace:
cd ~/PCC-Uspace/src
git checkout deep-learning
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/core/
echo $LD_LIBRARY_PATH



pip3 install --upgrade pip==19.0 
pip install tensorflow==1.15 numpy gym scipy stable-baselines[mpi]
Without MPI version of stable-baselines:
pip install tensorflow==1.15 numpy gym scipy stable-baselines


./app/pccserver recv 9000 &
./app/pccclient send 127.0.0.1 9000
./app/pccclient send 127.0.0.1 9000 --pcc-rate-control=python -pyhelper=loaded_client -pypath=~/PCC-RL/src/udt-plugins/testing/ --history-len=10 --pcc-utility-calc=linear --model-path=model_path_name
./app/pccclient send 127.0.0.1 9000 Vivace 
