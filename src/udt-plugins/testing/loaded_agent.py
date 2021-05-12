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

import tensorflow as tf
import numpy as np
import io
import sys

class LoadedModel():

    def __init__(self, model_path):
        self.sess = tf.Session()
        #print("self.sess: " + str(self.sess))
        self.model_path = model_path
        print("Model path set to " + str(model_path) + "\nNow, attempting to load the model...")
        self.metagraph = tf.saved_model.loader.load(self.sess,
            [tf.saved_model.tag_constants.SERVING], self.model_path)
        #print("self.metagraph: " + str(self.metagraph))
        sig = self.metagraph.signature_def["serving_default"]
        #print("Signature of the model: " + str(sig))
        input_dict = dict(sig.inputs)
        #print("input_dict of the model: " + str(input_dict))
        output_dict = dict(sig.outputs)    
        #print("output_dict of the model: " + str(output_dict))   
 
        self.input_obs_label = input_dict["ob"].name
        #print("input_obs_label: " + str(self.input_obs_label))
        self.input_state_label = None
        self.initial_state = None
        self.state = None
        if "state" in input_dict.keys():
            #replace looks fishy. Remove if it doesn't work.
            self.input_state_label = input_dict["state"].name.replace("train_model/", "")
            strfile = io.StringIO()
            #print(input_dict["state"].tensor_shape, file=strfile)
            #print("state: shape:")
            #print(input_dict["state"].tensor_shape)
            lines = strfile.getvalue().split("\n")
            dim_1 = int(lines[1].split(":")[1].strip(" "))
            dim_2 = int(lines[4].split(":")[1].strip(" "))
            print("dim_1: " + str(dim_1) + "\tdim_2: " + str(dim_2) + "\n")
            self.initial_state = np.zeros((dim_1, dim_2), dtype=np.float32)
            self.state = np.zeros((dim_1, dim_2), dtype=np.float32)
        #else:
            #print("Exiting because state couldn't be found with a Recurrent Policy in use.")
            #sys.exit()
 
        self.output_act_label = output_dict["act"].name
        #print("output_act_label: " + str(self.output_act_label))
        self.output_stochastic_act_label = None
        if "stochastic_act" in output_dict.keys():
            self.output_stochastic_act_label = output_dict["stochastic_act"].name
        #print("stochastic_act_label: " + str(self.output_stochastic_act_label))

        self.mask = None
        self.input_mask_label = None 
        if "mask" in input_dict.keys():
            #replace looks fishy. Remove if it doesn't work.
            self.input_mask_label = input_dict["mask"].name.replace("train_model/", "")
            #What to put here??????????????? None, 1, dynamic?????
            #self.mask = np.ones((1, 1)).reshape((1,))
            self.mask = np.ones(( 1 ))
        self.printed_input_dict = 0

    def reset_state(self):      
        self.state = np.copy(self.initial_state)

    def reload(self):
        self.metagraph = tf.saved_model.loader.load(self.sess,
            [tf.saved_model.tag_constants.SERVING], self.model_path)
 
    def act(self, obs, stochastic=False):
        #When applying RL to a custom problem, you should always normalize the input 
        #to the agent (e.g. using VecNormalizefor PPO2/A2C)
        #says https://readthedocs.org/projects/stable-baselines/downloads/pdf/master/ 
        input_dict = {self.input_obs_label:obs}
        #input_dict = {self.input_obs_label:obs}
        #print("in LoadedModel.act() input_dict: \n" + str(input_dict))
        if self.state is not None:
            input_dict[self.input_state_label] = self.state

        if self.mask is not None:
            input_dict[self.input_mask_label] = self.mask
        if(self.printed_input_dict == 0):
            #print("in LoadedModel.act() input_dict after state, mask added: \n" + str(input_dict))
            self.printed_input_dict = 1

        sess_output = None
        if stochastic:
            sess_output = self.sess.run(self.output_stochastic_act_label, feed_dict=input_dict)
        else:
            sess_output = self.sess.run(self.output_act_label, feed_dict=input_dict)

        action = None
        if len(sess_output) > 1:
            action, self.state = sess_output
        else:
            action = sess_output

        return {"act":action}


class LoadedModelAgent():

    def __init__(self, model_path):
        self.model = LoadedModel(model_path)

    def reset(self):
        self.model.reset_state()

    def act(self, ob):
        #stochastic = False???
        if(self.model.printed_input_dict == 0):
            print("observations: " + str(ob))
        act_dict = self.model.act(ob.reshape(1,-1), stochastic=False)

        ac = act_dict["act"]
        vpred = act_dict["vpred"] if "vpred" in act_dict.keys() else None
        state = act_dict["state"] if "state" in act_dict.keys() else None

        return ac[0][0]
