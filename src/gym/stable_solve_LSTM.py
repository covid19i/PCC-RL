# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import network_sim
import tensorflow as tf

from stable_baselines.common.policies import LstmPolicy
#to try after LstmPolicy
from stable_baselines.common.policies import MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO1, PPO2
import os
import sys
import inspect
from stable_baselines.common.env_checker import check_env

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.simple_arg_parse import arg_or_default

arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture for vf and pi is: %s" % str(arch))
#64, 64 is (sort of) the default for the shared layers in LstmPolicy according to 
#https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/policies.html#LstmPolicy
#Also, LSTMs are not supported in the shared part
net_arch = [64, 64, 'lstm', {"pi":arch, "vf":arch}]
lstm_dim = 64#no of parameters = 4(𝑛𝑚+𝑛^2+𝑛) = 33k for dim = 64 
print("Overall architecture is: %s" % str(net_arch))
print("LSTM dimenstion: %s" % str(lstm_dim))

training_sess = None

#https://stable-baselines.readthedocs.io/en/master/guide/examples.html#recurrent-policies
#One current limitation of recurrent policies is that you must test them with 
#the same number of environments they have been trained on.
#/home/ubuntu/environments/my_env/lib/python3.5/site-packages/stable_baselines/common/policies.py
#https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
class MyLstmPolicy(LstmPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=lstm_dim, reuse=False, **_kwargs):
        #layer_norm = True gives the following error:
        #Traceback (most recent call last):
        #File "stable_solve_LSTM.py", line 26, in <module>
        #from caffe2.python.helpers.normalization import layer_norm
        #ImportError: No module named 'caffe2'
        super(MyLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, 
                                          layers = None, net_arch=net_arch, layer_norm = False, 
                                          feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess

#https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=lstm_dim, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess


env = gym.make('PccNs-v0')
check_env(env)
#AttributeError: 'SimulatedNetworkEnv' object has no attribute 'num_envs'
#print("Number of environments used for training (env.num_envs): " + str(env.num_envs))
#env = gym.make('CartPole-v0')

gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
#https://github.com/hill-a/stable-baselines/#implemented-algorithms
#https://stable-baselines.readthedocs.io/en/master/guide/algos.html
#PPO1 can't be used with Recurrent
#In Algo 1 of paper: T is timesteps_per_actorbatch, M = optim_batchsize << NT.
#model = PPO1(MyLstmPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)

#nminibatches – (int) Number of training minibatches per update. For recurrent policies, 
#the number of environments run in parallel should be a multiple of nminibatches.
#https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html?highlight=ppo2
#don’t forget to take the hyperparameters from the RL zoo for continuousactions problems - https://readthedocs.org/projects/stable-baselines/downloads/pdf/master/
model = PPO2(MyLstmPolicy, env, verbose=1, nminibatches = 1, n_steps=20, gamma=gamma)


#Stable Baselines 3 tutorial
#https://github.com/araffin/rl-tutorial-jnrr19
for i in range(0, 6):
    with model.graph.as_default():                                                                   
        saver = tf.train.Saver()                                                     
        saver.save(training_sess, "./pcc_model_%d.ckpt" % i)
    model.learn(total_timesteps=(1600 * 410))

##
#   Save the model to the location specified below.
##
default_export_dir = "/home/ubuntu/models/latest_trained_model/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)
with model.graph.as_default():

    #https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ppo2/ppo2.html#PPO2
    pol = model.act_model#act_model

    obs_ph = pol.obs_ph
    act = pol.deterministic_action
    sampled_act = pol.action#??
    
    train = model.train_model
    #param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
    mask_ph = train.dones_ph#masks
    #param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
    states_ph = train.states_ph
    

    obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
    outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
    stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
    mask_ph_tensor_info = tf.saved_model.utils.build_tensor_info(mask_ph)
    #Looks fishy. REMOVE THIS IF IT DOESN'T WORK.
    #mask_ph_tensor_info.name = mask_ph_tensor_info.name.replace("train_model/", "")
    states_ph_tensor_info = tf.saved_model.utils.build_tensor_info(states_ph)
    #Looks fishy. REMOVE THIS IF IT DOESN'T WORK.
    #states_ph_tensor_info.name = states_ph_tensor_info.name.replace("train_model/", "")
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"ob":obs_input, "state": states_ph_tensor_info, "mask": mask_ph_tensor_info},
        outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    print("Signature saved: " + str(signature))

    #"""
    signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                     signature}

    model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    model_builder.add_meta_graph_and_variables(model.sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_map,
        clear_devices=True)
    model_builder.save(as_text=True)#as_text = True not to be done in Prod
    
env.close()
