
# Playing Atari with Deep Reinforcement Learning

Paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Walkthorugh: https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec

# Imports


```python
import numpy as np
import pandas as pd
import gym
import time
from copy import deepcopy

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import History
from keras.backend import tf as ktf
from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.utils.generic_utils import Progbar

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, Policy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint

from IPython import display
from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

%load_ext autoreload
%autoreload 2
```

    /home/nathan/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.



```python
# check our devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 10175638306536510897
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 494927872
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 13509914708254334763
    physical_device_desc: "device: 0, name: GeForce GTX 650, pci bus id: 0000:01:00.0, compute capability: 3.0"
    ]


# Environment

We use Gym's environments for now (specifically, the Atari environments).  Later, we'll be able to define our own environments using Gym's API, but for now we'll just use their out of the box options.

In the paper, they use the game `Breakout` with a slightly modified behavior that 'skips' every 4 frames.  It does this because they found that they didn't gain much improvement from having the DQN make an action for every frame, and instead found that having it make an action on every 4th frame (and just repeating that action for the 4 frames) was sufficient in this environment.  Gym offers this same version of the game, which can be made using `BreakoutDeterministic-v4`.


```python
env = gym.make('BreakoutDeterministic-v4') # try out different envs
env.reset()

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
```

# Agent

In order to implement a DQN, we need to start with an agent.  Instead of writing one from scratch (which is more tedious than actually difficult), we're going to adapt code from the `keras-rl` library (https://github.com/keras-rl/keras-rl).  Their DQN implementation is designed to match the same one demonstrated in the paper.  I've looked through the code (at this point quite extensively), and it's implemented the same way we would, so it's quite suitable.

## Visualizer Callback

The keras-rl default visualizer is iffy when using a server so, we'll disable the default visualizer, and create a slightly modified version that suits our purposes.  Essentially, we just get the environment to return the image of the game as an RGB numpy array, then use Matplotlib to plot it.

Visualization should only be used for debugging/demonstration.  Using visualization during training will slow it down drastically, so use with caution.


```python
class Visualizer(KerasCallback):
    def __init__(self, view_machine_mode=False, show_labels=True):
        self.machine_mode = view_machine_mode
        self.show_labels = show_labels
    
    def _set_env(self, env):
        self.env = env
        self.img = plt.imshow(self.env.render(mode='rgb_array')) # only call this once
        self.frame = 0
        plt.figure(figsize = (10,10))

    def on_action_end(self, action, logs):
        """ Render environment at the end of each action """
        img = self.env.render(mode='rgb_array')
        
        if self.machine_mode:
            # This lines allow us to see what the NN sees
            img = img[::2, ::2] # downscale
            img = np.mean(img, axis=2).astype(np.uint8) # grayscale
        
        self.frame += 1
        plt.cla()
        plt.imshow(img)
        if self.show_labels:
            plt.annotate('frame : ' + str(self.frame), xy=(10, 40), color='white')
            plt.annotate('action: ' + str(action), xy=(10, 50), color='white')
            
        display.display(plt.gcf())
        display.clear_output(wait=True)
```

## Preprocessor

Since the environment is pre-built and the agent is pre-built, we create a keras-rl Processor to treat data according to the paper's specifications (downsample and grayscale).

Note: we could modify the environment, the agent, or both instead to do this if we needed, but keras-rl provides this API to be able to avoid that.

https://github.com/keras-rl/keras-rl/blob/master/rl/core.py


```python
class AtariProcessor(Processor):      
    def process_observation(self, observation):
        # Normally, the observation is passed directly to the NN,
        # but we override this behavior to follow preprocessing steps
        img = observation 
        img = img[::2, ::2] # downscale
        img = np.mean(img, axis=2).astype(np.uint8) # grayscale
        return img
```

## Model

This is the model used in the paper (or as close to it as I could get).  It's a pretty straight-forward CNN.  It implements the input's normalization as a `Lambda` layer (so that we don't have to preprocess the data in that respect).

The input is actually 4 Grayscale frames, hence the input shape of `(4, 105, 80)`.  This is how it was designed in the paper (it uses the 4 most recent frames so that the DQN can figure out things like the ball's trajectory).

We also don't directly compile the model.  The keras-rl library takes care of this for us, which is important since it handles 'special' properties of the model, such as using the **Huber Loss** and the output masking.  It should be noted that we could implement these things pretty readily in the model definition (as the walkthrough describes), but we don't since we're relying on keras-rl's implementation.


```python
input_shape = (4, 105, 80)

frames_input = keras.layers.Input(input_shape, name='frames')

# performs normalization directly in model
normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

conv_1 = keras.layers.Conv2D(32, (8,8), strides=(4, 4), activation='relu', data_format='channels_first')(normalized)
conv_2 = keras.layers.Conv2D(64, (4,4), strides=(2, 2), activation='relu')(conv_1)
conv_3 = keras.layers.Conv2D(64, (3,3), strides=(1, 1), activation='relu')(conv_2)

conv_flattened = keras.layers.Flatten()(conv_3)
hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
output = keras.layers.Dense(nb_actions)(hidden)

model = keras.models.Model(inputs=frames_input, outputs=output)
optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
#model.compile(optimizer, loss='mse')
```

## Agent Creation

Finally, we put everything together to create the agent.  We need three things to actually create the agent: it's memory, it's policy, and it's model.  We created the model above, and keras-rl provides us with suitable classes for memory and policy.

The memory we'll be using is `Sequential Memory`, which works as described in the paper.  Note that we're using a `window_length` of 4, to reflect the 4 frames of information we need as described before.

The policy we'll be using for training is `EpsGreedyQPolicy` setup with `LinearAnnealedPolicy`.  The `EpsGreedyPolicy` is the ε-Greedy policy as described in the paper (it chooses a random action with a proportionate frequency to ε).  The `LinearAnnealedPolicy` is a class wrapper that anneals the ε value in the ε-Greedy policy (as described in the paper).

Finally, we put all this together in an instance of the DQNAgent class.  We pass through all of it's components, then compile it.


```python
memory = SequentialMemory(limit=500000, window_length=4)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 1., 0.1, 0, 100000)
dqn = DQNAgent(model=model, processor=AtariProcessor(), nb_actions=nb_actions, 
               memory=memory, nb_steps_warmup=50, target_model_update=1e-2, policy=policy)
dqn.compile(optimizer, metrics=['mse'])
```

# Training

Finally, we can train the thing.  As mentioned above, we can add the `Visualizer` callback to the `fit` method to plot the image and set `visualize` to `False`.  We set `action_repetition` to 4 so that the agent only acts on every 4th frame.  The model's `fit` method returns the training history, so we can look at that later by saving to the `hist` variable.

After training has completed, we save the weights of the model (note: this will overwrite previous models.  If you want to save previous models, either move them or save the new one to a different path).  We also save the history to a pickle file for later viewing.


```python
callback_list = []
train_name = 'dqn_' + str(int(time.time()))
model_path = "sessions/" + train_name + "/model.hdf5"
log_path = "sessions/" + train_name +"/log.csv"
tensorboard_path = "sessions/" + train_name +"/" + train_name

callback_list.append(TensorBoard(log_dir=tensorboard_path))
callback_list.append(ModelIntervalCheckpoint(model_path, 500))
#callback_list.append(CSVLogger(log_path, append=True))
#callback_list.append(Visualizer())

hist = dqn.fit(env, nb_steps=500000, visualize=False, action_repetition=4,
               callbacks=callback_list, verbose=2)

# After training is done, we save the final weights
dqn.save_weights('dqn_{}_weights.h5f'.format('test'), overwrite=True)

# and save the history to a pickle file
results = pd.DataFrame(hist.history)
results.to_pickle("sessions/" + train_name +"/hist.pickle")
```

    Training for 500000 steps ...
         70/500000: episode: 1, duration: 2.488s, episode steps: 70, steps per second: 28, episode reward: 4.000, mean reward: 0.057 [0.000, 1.000], mean action: 1.500 [0.000, 3.000], mean observation: 39.597 [0.000, 142.000], loss: 0.019269, mean_squared_error: 0.011260, mean_q: 0.061806, mean_eps: 0.999460
        133/500000: episode: 2, duration: 6.458s, episode steps: 63, steps per second: 10, episode reward: 3.000, mean reward: 0.048 [0.000, 1.000], mean action: 1.444 [0.000, 3.000], mean observation: 39.692 [0.000, 142.000], loss: 0.018505, mean_squared_error: 0.012372, mean_q: 0.084370, mean_eps: 0.999091
        197/500000: episode: 3, duration: 6.546s, episode steps: 64, steps per second: 10, episode reward: 2.000, mean reward: 0.031 [0.000, 1.000], mean action: 1.547 [0.000, 3.000], mean observation: 39.732 [0.000, 142.000], loss: 0.017170, mean_squared_error: 0.015053, mean_q: 0.110219, mean_eps: 0.998520
        248/500000: episode: 4, duration: 5.281s, episode steps: 51, steps per second: 10, episode reward: 1.000, mean reward: 0.020 [0.000, 1.000], mean action: 1.451 [0.000, 3.000], mean observation: 39.717 [0.000, 142.000], loss: 0.017020, mean_squared_error: 0.018690, mean_q: 0.137555, mean_eps: 0.998002
        298/500000: episode: 5, duration: 5.233s, episode steps: 50, steps per second: 10, episode reward: 1.000, mean reward: 0.020 [0.000, 1.000], mean action: 1.680 [0.000, 3.000], mean observation: 39.746 [0.000, 142.000], loss: 0.019283, mean_squared_error: 0.024505, mean_q: 0.159967, mean_eps: 0.997548
        360/500000: episode: 6, duration: 6.370s, episode steps: 62, steps per second: 10, episode reward: 1.000, mean reward: 0.016 [0.000, 1.000], mean action: 1.548 [0.000, 3.000], mean observation: 39.972 [0.000, 142.000], loss: 0.015923, mean_squared_error: 0.028140, mean_q: 0.182266, mean_eps: 0.997044
        446/500000: episode: 7, duration: 8.976s, episode steps: 86, steps per second: 10, episode reward: 3.000, mean reward: 0.035 [0.000, 1.000], mean action: 1.558 [0.000, 3.000], mean observation: 39.625 [0.000, 142.000], loss: 0.017484, mean_squared_error: 0.033623, mean_q: 0.194334, mean_eps: 0.996377
        496/500000: episode: 8, duration: 5.208s, episode steps: 50, steps per second: 10, episode reward: 1.000, mean reward: 0.020 [0.000, 1.000], mean action: 1.300 [0.000, 3.000], mean observation: 39.794 [0.000, 142.000], loss: 0.016503, mean_squared_error: 0.039375, mean_q: 0.215635, mean_eps: 0.995765
        571/500000: episode: 9, duration: 7.735s, episode steps: 75, steps per second: 10, episode reward: 3.000, mean reward: 0.040 [0.000, 1.000], mean action: 1.733 [0.000, 3.000], mean observation: 39.734 [0.000, 142.000], loss: 0.014402, mean_squared_error: 0.041402, mean_q: 0.227490, mean_eps: 0.995203
        619/500000: episode: 10, duration: 4.880s, episode steps: 48, steps per second: 10, episode reward: 1.000, mean reward: 0.021 [0.000, 1.000], mean action: 1.188 [0.000, 3.000], mean observation: 39.680 [0.000, 142.000], loss: 0.015515, mean_squared_error: 0.048263, mean_q: 0.245671, mean_eps: 0.994649
        682/500000: episode: 11, duration: 6.390s, episode steps: 63, steps per second: 10, episode reward: 0.000, mean reward: 0.000 [0.000, 0.000], mean action: 1.698 [0.000, 3.000], mean observation: 39.996 [0.000, 142.000], loss: 0.017431, mean_squared_error: 0.057560, mean_q: 0.272968, mean_eps: 0.994150
        747/500000: episode: 12, duration: 6.650s, episode steps: 65, steps per second: 10, episode reward: 3.000, mean reward: 0.046 [0.000, 1.000], mean action: 1.492 [0.000, 3.000], mean observation: 39.659 [0.000, 142.000], loss: 0.016893, mean_squared_error: 0.064775, mean_q: 0.288614, mean_eps: 0.993574
        802/500000: episode: 13, duration: 5.614s, episode steps: 55, steps per second: 10, episode reward: 0.000, mean reward: 0.000 [0.000, 0.000], mean action: 1.691 [0.000, 3.000], mean observation: 40.028 [0.000, 142.000], loss: 0.015650, mean_squared_error: 0.072341, mean_q: 0.306288, mean_eps: 0.993034
        855/500000: episode: 14, duration: 5.332s, episode steps: 53, steps per second: 10, episode reward: 1.000, mean reward: 0.019 [0.000, 1.000], mean action: 1.642 [0.000, 3.000], mean observation: 39.778 [0.000, 142.000], loss: 0.017147, mean_squared_error: 0.076925, mean_q: 0.316564, mean_eps: 0.992548
        932/500000: episode: 15, duration: 7.685s, episode steps: 77, steps per second: 10, episode reward: 4.000, mean reward: 0.052 [0.000, 1.000], mean action: 1.299 [0.000, 3.000], mean observation: 39.554 [0.000, 142.000], loss: 0.013175, mean_squared_error: 0.081898, mean_q: 0.334115, mean_eps: 0.991963
        990/500000: episode: 16, duration: 5.853s, episode steps: 58, steps per second: 10, episode reward: 1.000, mean reward: 0.017 [0.000, 1.000], mean action: 1.328 [0.000, 3.000], mean observation: 39.701 [0.000, 142.000], loss: 0.015155, mean_squared_error: 0.091665, mean_q: 0.344775, mean_eps: 0.991355
       1042/500000: episode: 17, duration: 5.257s, episode steps: 52, steps per second: 10, episode reward: 1.000, mean reward: 0.019 [0.000, 1.000], mean action: 1.577 [0.000, 3.000], mean observation: 39.901 [0.000, 142.000], loss: 0.016044, mean_squared_error: 0.100769, mean_q: 0.372955, mean_eps: 0.990861
       1117/500000: episode: 18, duration: 7.584s, episode steps: 75, steps per second: 10, episode reward: 4.000, mean reward: 0.053 [0.000, 1.000], mean action: 1.427 [0.000, 3.000], mean observation: 39.586 [0.000, 142.000], loss: 0.014454, mean_squared_error: 0.108404, mean_q: 0.392218, mean_eps: 0.990289
       1200/500000: episode: 19, duration: 8.325s, episode steps: 83, steps per second: 10, episode reward: 3.000, mean reward: 0.036 [0.000, 1.000], mean action: 1.494 [0.000, 3.000], mean observation: 39.684 [0.000, 142.000], loss: 0.016304, mean_squared_error: 0.120514, mean_q: 0.411272, mean_eps: 0.989578
      91630/500000: episode: 1286, duration: 8.504s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.774 [0.000, 3.000], mean observation: 39.515 [0.000, 142.000], loss: 0.030447, mean_squared_error: 29.281305, mean_q: 5.895939, mean_eps: 0.175712
      91724/500000: episode: 1287, duration: 9.560s, episode steps: 94, steps per second: 10, episode reward: 13.000, mean reward: 0.138 [0.000, 4.000], mean action: 1.638 [0.000, 3.000], mean observation: 39.411 [0.000, 142.000], loss: 0.032840, mean_squared_error: 29.700829, mean_q: 5.963631, mean_eps: 0.174911
      91813/500000: episode: 1288, duration: 8.959s, episode steps: 89, steps per second: 10, episode reward: 9.000, mean reward: 0.101 [0.000, 4.000], mean action: 2.067 [0.000, 3.000], mean observation: 39.520 [0.000, 142.000], loss: 0.027671, mean_squared_error: 30.141624, mean_q: 6.022346, mean_eps: 0.174088
      91899/500000: episode: 1289, duration: 8.736s, episode steps: 86, steps per second: 10, episode reward: 12.000, mean reward: 0.140 [0.000, 4.000], mean action: 1.709 [0.000, 3.000], mean observation: 39.526 [0.000, 142.000], loss: 0.029749, mean_squared_error: 29.547089, mean_q: 5.935153, mean_eps: 0.173300
      91995/500000: episode: 1290, duration: 9.736s, episode steps: 96, steps per second: 10, episode reward: 6.000, mean reward: 0.062 [0.000, 1.000], mean action: 1.458 [0.000, 3.000], mean observation: 39.512 [0.000, 142.000], loss: 0.032223, mean_squared_error: 30.198047, mean_q: 6.006385, mean_eps: 0.172481
      92074/500000: episode: 1291, duration: 8.026s, episode steps: 79, steps per second: 10, episode reward: 8.000, mean reward: 0.101 [0.000, 4.000], mean action: 1.899 [0.000, 3.000], mean observation: 39.588 [0.000, 142.000], loss: 0.035990, mean_squared_error: 30.203020, mean_q: 6.023508, mean_eps: 0.171694
      92155/500000: episode: 1292, duration: 8.251s, episode steps: 81, steps per second: 10, episode reward: 8.000, mean reward: 0.099 [0.000, 4.000], mean action: 1.593 [0.000, 3.000], mean observation: 39.565 [0.000, 142.000], loss: 0.031128, mean_squared_error: 30.970011, mean_q: 6.118612, mean_eps: 0.170974
      92246/500000: episode: 1293, duration: 9.238s, episode steps: 91, steps per second: 10, episode reward: 9.000, mean reward: 0.099 [0.000, 4.000], mean action: 1.912 [0.000, 3.000], mean observation: 39.472 [0.000, 142.000], loss: 0.032136, mean_squared_error: 30.631103, mean_q: 6.079716, mean_eps: 0.170200
      92369/500000: episode: 1294, duration: 12.489s, episode steps: 123, steps per second: 10, episode reward: 13.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.894 [0.000, 3.000], mean observation: 39.158 [0.000, 142.000], loss: 0.034901, mean_squared_error: 30.242414, mean_q: 6.026470, mean_eps: 0.169237
      92462/500000: episode: 1295, duration: 9.439s, episode steps: 93, steps per second: 10, episode reward: 10.000, mean reward: 0.108 [0.000, 4.000], mean action: 1.817 [0.000, 3.000], mean observation: 39.433 [0.000, 142.000], loss: 0.034058, mean_squared_error: 30.443722, mean_q: 6.036304, mean_eps: 0.168265
      92542/500000: episode: 1296, duration: 8.212s, episode steps: 80, steps per second: 10, episode reward: 8.000, mean reward: 0.100 [0.000, 4.000], mean action: 1.837 [0.000, 3.000], mean observation: 39.596 [0.000, 142.000], loss: 0.034660, mean_squared_error: 29.826434, mean_q: 5.995749, mean_eps: 0.167486
      92632/500000: episode: 1297, duration: 9.122s, episode steps: 90, steps per second: 10, episode reward: 9.000, mean reward: 0.100 [0.000, 4.000], mean action: 1.856 [0.000, 3.000], mean observation: 39.506 [0.000, 142.000], loss: 0.036109, mean_squared_error: 30.112514, mean_q: 6.020973, mean_eps: 0.166721
      92716/500000: episode: 1298, duration: 8.547s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 2.036 [0.000, 3.000], mean observation: 39.512 [0.000, 142.000], loss: 0.030358, mean_squared_error: 30.075096, mean_q: 6.028612, mean_eps: 0.165938
      92800/500000: episode: 1299, duration: 8.554s, episode steps: 84, steps per second: 10, episode reward: 9.000, mean reward: 0.107 [0.000, 4.000], mean action: 1.619 [0.000, 3.000], mean observation: 39.487 [0.000, 142.000], loss: 0.033073, mean_squared_error: 30.197228, mean_q: 6.006042, mean_eps: 0.165183
      92869/500000: episode: 1300, duration: 6.973s, episode steps: 69, steps per second: 10, episode reward: 7.000, mean reward: 0.101 [0.000, 4.000], mean action: 1.855 [0.000, 3.000], mean observation: 39.613 [0.000, 142.000], loss: 0.033015, mean_squared_error: 30.110901, mean_q: 6.019593, mean_eps: 0.164494
      92954/500000: episode: 1301, duration: 8.536s, episode steps: 85, steps per second: 10, episode reward: 12.000, mean reward: 0.141 [0.000, 4.000], mean action: 1.659 [0.000, 3.000], mean observation: 39.523 [0.000, 142.000], loss: 0.033424, mean_squared_error: 30.691437, mean_q: 6.087956, mean_eps: 0.163801
      93041/500000: episode: 1302, duration: 8.906s, episode steps: 87, steps per second: 10, episode reward: 9.000, mean reward: 0.103 [0.000, 4.000], mean action: 2.023 [0.000, 3.000], mean observation: 39.497 [0.000, 142.000], loss: 0.036913, mean_squared_error: 30.227067, mean_q: 6.034699, mean_eps: 0.163027
      93120/500000: episode: 1303, duration: 8.069s, episode steps: 79, steps per second: 10, episode reward: 8.000, mean reward: 0.101 [0.000, 4.000], mean action: 1.646 [0.000, 3.000], mean observation: 39.589 [0.000, 142.000], loss: 0.033593, mean_squared_error: 29.529240, mean_q: 5.946997, mean_eps: 0.162280
      93203/500000: episode: 1304, duration: 8.450s, episode steps: 83, steps per second: 10, episode reward: 12.000, mean reward: 0.145 [0.000, 4.000], mean action: 1.735 [0.000, 3.000], mean observation: 39.516 [0.000, 142.000], loss: 0.033077, mean_squared_error: 30.050201, mean_q: 5.998655, mean_eps: 0.161551
      93295/500000: episode: 1305, duration: 9.360s, episode steps: 92, steps per second: 10, episode reward: 9.000, mean reward: 0.098 [0.000, 4.000], mean action: 1.978 [0.000, 3.000], mean observation: 39.521 [0.000, 142.000], loss: 0.033772, mean_squared_error: 30.044353, mean_q: 6.025092, mean_eps: 0.160763
      93391/500000: episode: 1306, duration: 9.805s, episode steps: 96, steps per second: 10, episode reward: 13.000, mean reward: 0.135 [0.000, 4.000], mean action: 1.740 [0.000, 3.000], mean observation: 39.414 [0.000, 142.000], loss: 0.030073, mean_squared_error: 30.351744, mean_q: 6.030584, mean_eps: 0.159917
      93508/500000: episode: 1307, duration: 11.825s, episode steps: 117, steps per second: 10, episode reward: 11.000, mean reward: 0.094 [0.000, 4.000], mean action: 1.641 [0.000, 3.000], mean observation: 39.273 [0.000, 142.000], loss: 0.028755, mean_squared_error: 30.126919, mean_q: 6.015755, mean_eps: 0.158959
      93595/500000: episode: 1308, duration: 8.757s, episode steps: 87, steps per second: 10, episode reward: 12.000, mean reward: 0.138 [0.000, 4.000], mean action: 1.621 [0.000, 3.000], mean observation: 39.524 [0.000, 142.000], loss: 0.028510, mean_squared_error: 30.779583, mean_q: 6.114129, mean_eps: 0.158041
      93698/500000: episode: 1309, duration: 10.346s, episode steps: 103, steps per second: 10, episode reward: 11.000, mean reward: 0.107 [0.000, 4.000], mean action: 1.903 [0.000, 3.000], mean observation: 39.287 [0.000, 142.000], loss: 0.031841, mean_squared_error: 29.977223, mean_q: 6.006899, mean_eps: 0.157186
      93770/500000: episode: 1310, duration: 7.274s, episode steps: 72, steps per second: 10, episode reward: 11.000, mean reward: 0.153 [0.000, 4.000], mean action: 1.708 [0.000, 3.000], mean observation: 39.549 [0.000, 142.000], loss: 0.032176, mean_squared_error: 30.424012, mean_q: 6.073835, mean_eps: 0.156398
      93851/500000: episode: 1311, duration: 8.277s, episode steps: 81, steps per second: 10, episode reward: 5.000, mean reward: 0.062 [0.000, 1.000], mean action: 1.889 [0.000, 3.000], mean observation: 39.556 [0.000, 142.000], loss: 0.030424, mean_squared_error: 29.741918, mean_q: 5.958858, mean_eps: 0.155710
      93958/500000: episode: 1312, duration: 10.889s, episode steps: 107, steps per second: 10, episode reward: 8.000, mean reward: 0.075 [0.000, 1.000], mean action: 1.252 [0.000, 3.000], mean observation: 39.319 [0.000, 142.000], loss: 0.030675, mean_squared_error: 31.199331, mean_q: 6.157454, mean_eps: 0.154864
      94035/500000: episode: 1313, duration: 7.842s, episode steps: 77, steps per second: 10, episode reward: 8.000, mean reward: 0.104 [0.000, 4.000], mean action: 1.727 [0.000, 3.000], mean observation: 39.559 [0.000, 142.000], loss: 0.034464, mean_squared_error: 30.521927, mean_q: 6.070559, mean_eps: 0.154036
      94133/500000: episode: 1314, duration: 9.961s, episode steps: 98, steps per second: 10, episode reward: 13.000, mean reward: 0.133 [0.000, 4.000], mean action: 1.684 [0.000, 3.000], mean observation: 39.413 [0.000, 142.000], loss: 0.030873, mean_squared_error: 30.519856, mean_q: 6.054660, mean_eps: 0.153248
      94234/500000: episode: 1315, duration: 10.254s, episode steps: 101, steps per second: 10, episode reward: 10.000, mean reward: 0.099 [0.000, 4.000], mean action: 1.832 [0.000, 3.000], mean observation: 39.402 [0.000, 142.000], loss: 0.035975, mean_squared_error: 30.052239, mean_q: 5.983097, mean_eps: 0.152353
      94324/500000: episode: 1316, duration: 9.087s, episode steps: 90, steps per second: 10, episode reward: 9.000, mean reward: 0.100 [0.000, 4.000], mean action: 1.856 [0.000, 3.000], mean observation: 39.543 [0.000, 142.000], loss: 0.028134, mean_squared_error: 30.730821, mean_q: 6.104345, mean_eps: 0.151493
      94409/500000: episode: 1317, duration: 8.639s, episode steps: 85, steps per second: 10, episode reward: 9.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.612 [0.000, 3.000], mean observation: 39.510 [0.000, 142.000], loss: 0.033556, mean_squared_error: 30.847120, mean_q: 6.095350, mean_eps: 0.150706
      94496/500000: episode: 1318, duration: 8.862s, episode steps: 87, steps per second: 10, episode reward: 12.000, mean reward: 0.138 [0.000, 4.000], mean action: 1.931 [0.000, 3.000], mean observation: 39.522 [0.000, 142.000], loss: 0.028890, mean_squared_error: 30.712723, mean_q: 6.087029, mean_eps: 0.149932
      94596/500000: episode: 1319, duration: 10.150s, episode steps: 100, steps per second: 10, episode reward: 10.000, mean reward: 0.100 [0.000, 4.000], mean action: 2.080 [0.000, 3.000], mean observation: 39.405 [0.000, 142.000], loss: 0.030806, mean_squared_error: 30.369680, mean_q: 6.043041, mean_eps: 0.149090
      94685/500000: episode: 1320, duration: 8.988s, episode steps: 89, steps per second: 10, episode reward: 9.000, mean reward: 0.101 [0.000, 4.000], mean action: 1.528 [0.000, 3.000], mean observation: 39.472 [0.000, 142.000], loss: 0.032239, mean_squared_error: 30.888201, mean_q: 6.101109, mean_eps: 0.148240
      94761/500000: episode: 1321, duration: 7.761s, episode steps: 76, steps per second: 10, episode reward: 11.000, mean reward: 0.145 [0.000, 4.000], mean action: 1.829 [0.000, 3.000], mean observation: 39.557 [0.000, 142.000], loss: 0.033658, mean_squared_error: 31.085387, mean_q: 6.137527, mean_eps: 0.147497
      94867/500000: episode: 1322, duration: 10.817s, episode steps: 106, steps per second: 10, episode reward: 11.000, mean reward: 0.104 [0.000, 4.000], mean action: 1.925 [0.000, 3.000], mean observation: 39.264 [0.000, 142.000], loss: 0.032383, mean_squared_error: 30.905907, mean_q: 6.105449, mean_eps: 0.146678
      94951/500000: episode: 1323, duration: 8.487s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.738 [0.000, 3.000], mean observation: 39.492 [0.000, 142.000], loss: 0.034475, mean_squared_error: 30.684495, mean_q: 6.084649, mean_eps: 0.145823
      95036/500000: episode: 1324, duration: 8.658s, episode steps: 85, steps per second: 10, episode reward: 9.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.647 [0.000, 3.000], mean observation: 39.487 [0.000, 142.000], loss: 0.026047, mean_squared_error: 31.287795, mean_q: 6.150253, mean_eps: 0.145063
      95109/500000: episode: 1325, duration: 7.412s, episode steps: 73, steps per second: 10, episode reward: 11.000, mean reward: 0.151 [0.000, 4.000], mean action: 2.014 [0.000, 3.000], mean observation: 39.553 [0.000, 142.000], loss: 0.031814, mean_squared_error: 30.277339, mean_q: 6.027421, mean_eps: 0.144352
      95184/500000: episode: 1326, duration: 7.597s, episode steps: 75, steps per second: 10, episode reward: 11.000, mean reward: 0.147 [0.000, 4.000], mean action: 1.720 [0.000, 3.000], mean observation: 39.543 [0.000, 142.000], loss: 0.033027, mean_squared_error: 29.907761, mean_q: 5.968772, mean_eps: 0.143686
      95283/500000: episode: 1327, duration: 10.031s, episode steps: 99, steps per second: 10, episode reward: 13.000, mean reward: 0.131 [0.000, 4.000], mean action: 1.899 [0.000, 3.000], mean observation: 39.432 [0.000, 142.000], loss: 0.034350, mean_squared_error: 30.540784, mean_q: 6.039264, mean_eps: 0.142903
      95370/500000: episode: 1328, duration: 8.795s, episode steps: 87, steps per second: 10, episode reward: 9.000, mean reward: 0.103 [0.000, 4.000], mean action: 1.736 [0.000, 3.000], mean observation: 39.505 [0.000, 142.000], loss: 0.027672, mean_squared_error: 30.566773, mean_q: 6.039312, mean_eps: 0.142066
      95468/500000: episode: 1329, duration: 9.980s, episode steps: 98, steps per second: 10, episode reward: 10.000, mean reward: 0.102 [0.000, 4.000], mean action: 1.786 [0.000, 3.000], mean observation: 39.421 [0.000, 142.000], loss: 0.038724, mean_squared_error: 30.882565, mean_q: 6.103698, mean_eps: 0.141233
      95556/500000: episode: 1330, duration: 8.960s, episode steps: 88, steps per second: 10, episode reward: 5.000, mean reward: 0.057 [0.000, 1.000], mean action: 1.636 [0.000, 3.000], mean observation: 39.531 [0.000, 142.000], loss: 0.031726, mean_squared_error: 30.759808, mean_q: 6.064096, mean_eps: 0.140396
      95634/500000: episode: 1331, duration: 7.947s, episode steps: 78, steps per second: 10, episode reward: 8.000, mean reward: 0.103 [0.000, 4.000], mean action: 1.474 [0.000, 3.000], mean observation: 39.584 [0.000, 142.000], loss: 0.029391, mean_squared_error: 30.616167, mean_q: 6.041346, mean_eps: 0.139649
      95732/500000: episode: 1332, duration: 9.926s, episode steps: 98, steps per second: 10, episode reward: 10.000, mean reward: 0.102 [0.000, 4.000], mean action: 1.796 [0.000, 3.000], mean observation: 39.389 [0.000, 142.000], loss: 0.029753, mean_squared_error: 30.488569, mean_q: 6.053084, mean_eps: 0.138857
      95817/500000: episode: 1333, duration: 8.633s, episode steps: 85, steps per second: 10, episode reward: 5.000, mean reward: 0.059 [0.000, 1.000], mean action: 1.741 [0.000, 3.000], mean observation: 39.561 [0.000, 142.000], loss: 0.027888, mean_squared_error: 30.948763, mean_q: 6.097194, mean_eps: 0.138034
      95896/500000: episode: 1334, duration: 8.023s, episode steps: 79, steps per second: 10, episode reward: 8.000, mean reward: 0.101 [0.000, 4.000], mean action: 1.671 [0.000, 3.000], mean observation: 39.580 [0.000, 142.000], loss: 0.029716, mean_squared_error: 30.901267, mean_q: 6.087179, mean_eps: 0.137296
      95952/500000: episode: 1335, duration: 5.741s, episode steps: 56, steps per second: 10, episode reward: 3.000, mean reward: 0.054 [0.000, 1.000], mean action: 1.607 [0.000, 3.000], mean observation: 39.716 [0.000, 142.000], loss: 0.031209, mean_squared_error: 31.490863, mean_q: 6.154454, mean_eps: 0.136688
      96030/500000: episode: 1336, duration: 7.947s, episode steps: 78, steps per second: 10, episode reward: 8.000, mean reward: 0.103 [0.000, 4.000], mean action: 1.846 [0.000, 3.000], mean observation: 39.619 [0.000, 142.000], loss: 0.033327, mean_squared_error: 30.911434, mean_q: 6.101139, mean_eps: 0.136085
      96132/500000: episode: 1337, duration: 10.298s, episode steps: 102, steps per second: 10, episode reward: 11.000, mean reward: 0.108 [0.000, 4.000], mean action: 1.843 [0.000, 3.000], mean observation: 39.328 [0.000, 142.000], loss: 0.032135, mean_squared_error: 30.800145, mean_q: 6.095021, mean_eps: 0.135276
      96217/500000: episode: 1338, duration: 8.627s, episode steps: 85, steps per second: 10, episode reward: 12.000, mean reward: 0.141 [0.000, 4.000], mean action: 1.765 [0.000, 3.000], mean observation: 39.522 [0.000, 142.000], loss: 0.033408, mean_squared_error: 31.060833, mean_q: 6.098504, mean_eps: 0.134434
      96301/500000: episode: 1339, duration: 8.524s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.821 [0.000, 3.000], mean observation: 39.507 [0.000, 142.000], loss: 0.034035, mean_squared_error: 30.939524, mean_q: 6.099551, mean_eps: 0.133673
      96406/500000: episode: 1340, duration: 10.667s, episode steps: 105, steps per second: 10, episode reward: 14.000, mean reward: 0.133 [0.000, 4.000], mean action: 1.648 [0.000, 3.000], mean observation: 39.346 [0.000, 142.000], loss: 0.031040, mean_squared_error: 31.337325, mean_q: 6.154335, mean_eps: 0.132823
      96507/500000: episode: 1341, duration: 10.260s, episode steps: 101, steps per second: 10, episode reward: 11.000, mean reward: 0.109 [0.000, 4.000], mean action: 1.881 [0.000, 3.000], mean observation: 39.306 [0.000, 142.000], loss: 0.032674, mean_squared_error: 30.562558, mean_q: 6.067238, mean_eps: 0.131896
      96599/500000: episode: 1342, duration: 9.385s, episode steps: 92, steps per second: 10, episode reward: 13.000, mean reward: 0.141 [0.000, 4.000], mean action: 2.011 [0.000, 3.000], mean observation: 39.404 [0.000, 142.000], loss: 0.034983, mean_squared_error: 31.423469, mean_q: 6.162767, mean_eps: 0.131027
      96681/500000: episode: 1343, duration: 8.350s, episode steps: 82, steps per second: 10, episode reward: 5.000, mean reward: 0.061 [0.000, 1.000], mean action: 1.683 [0.000, 3.000], mean observation: 39.580 [0.000, 142.000], loss: 0.031190, mean_squared_error: 30.886762, mean_q: 6.108385, mean_eps: 0.130244
      96773/500000: episode: 1344, duration: 9.346s, episode steps: 92, steps per second: 10, episode reward: 6.000, mean reward: 0.065 [0.000, 1.000], mean action: 1.728 [0.000, 3.000], mean observation: 39.510 [0.000, 142.000], loss: 0.038428, mean_squared_error: 30.654394, mean_q: 6.056853, mean_eps: 0.129461
      96894/500000: episode: 1345, duration: 12.214s, episode steps: 121, steps per second: 10, episode reward: 15.000, mean reward: 0.124 [0.000, 4.000], mean action: 2.091 [0.000, 3.000], mean observation: 39.170 [0.000, 142.000], loss: 0.032677, mean_squared_error: 31.230785, mean_q: 6.145153, mean_eps: 0.128503
      96981/500000: episode: 1346, duration: 8.896s, episode steps: 87, steps per second: 10, episode reward: 9.000, mean reward: 0.103 [0.000, 4.000], mean action: 1.920 [0.000, 3.000], mean observation: 39.495 [0.000, 142.000], loss: 0.030187, mean_squared_error: 31.228692, mean_q: 6.132100, mean_eps: 0.127567
      97053/500000: episode: 1347, duration: 7.350s, episode steps: 72, steps per second: 10, episode reward: 11.000, mean reward: 0.153 [0.000, 4.000], mean action: 2.000 [0.000, 3.000], mean observation: 39.542 [0.000, 142.000], loss: 0.034369, mean_squared_error: 30.657571, mean_q: 6.064613, mean_eps: 0.126851
      97125/500000: episode: 1348, duration: 7.381s, episode steps: 72, steps per second: 10, episode reward: 11.000, mean reward: 0.153 [0.000, 4.000], mean action: 1.417 [0.000, 3.000], mean observation: 39.544 [0.000, 142.000], loss: 0.030931, mean_squared_error: 31.654602, mean_q: 6.173853, mean_eps: 0.126203
      97226/500000: episode: 1349, duration: 10.253s, episode steps: 101, steps per second: 10, episode reward: 10.000, mean reward: 0.099 [0.000, 4.000], mean action: 1.802 [0.000, 3.000], mean observation: 39.422 [0.000, 142.000], loss: 0.031234, mean_squared_error: 31.336826, mean_q: 6.137400, mean_eps: 0.125425
      97293/500000: episode: 1350, duration: 6.806s, episode steps: 67, steps per second: 10, episode reward: 7.000, mean reward: 0.104 [0.000, 4.000], mean action: 2.015 [0.000, 3.000], mean observation: 39.570 [0.000, 142.000], loss: 0.032595, mean_squared_error: 30.980285, mean_q: 6.111974, mean_eps: 0.124669
      97377/500000: episode: 1351, duration: 8.566s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.786 [0.000, 3.000], mean observation: 39.508 [0.000, 142.000], loss: 0.029847, mean_squared_error: 31.833358, mean_q: 6.199978, mean_eps: 0.123989
      97471/500000: episode: 1352, duration: 9.535s, episode steps: 94, steps per second: 10, episode reward: 10.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.713 [0.000, 3.000], mean observation: 39.387 [0.000, 142.000], loss: 0.032440, mean_squared_error: 31.332655, mean_q: 6.122754, mean_eps: 0.123188
      97566/500000: episode: 1353, duration: 9.699s, episode steps: 95, steps per second: 10, episode reward: 13.000, mean reward: 0.137 [0.000, 4.000], mean action: 1.632 [0.000, 3.000], mean observation: 39.431 [0.000, 142.000], loss: 0.035603, mean_squared_error: 31.578970, mean_q: 6.161292, mean_eps: 0.122338
      97651/500000: episode: 1354, duration: 8.647s, episode steps: 85, steps per second: 10, episode reward: 9.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.435 [0.000, 3.000], mean observation: 39.518 [0.000, 142.000], loss: 0.027499, mean_squared_error: 32.471549, mean_q: 6.259760, mean_eps: 0.121528
      97735/500000: episode: 1355, duration: 8.570s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.821 [0.000, 3.000], mean observation: 39.507 [0.000, 142.000], loss: 0.034582, mean_squared_error: 31.307583, mean_q: 6.128037, mean_eps: 0.120767
      97812/500000: episode: 1356, duration: 7.849s, episode steps: 77, steps per second: 10, episode reward: 8.000, mean reward: 0.104 [0.000, 4.000], mean action: 2.013 [0.000, 3.000], mean observation: 39.577 [0.000, 142.000], loss: 0.032826, mean_squared_error: 31.536177, mean_q: 6.154376, mean_eps: 0.120043
      97908/500000: episode: 1357, duration: 9.642s, episode steps: 96, steps per second: 10, episode reward: 13.000, mean reward: 0.135 [0.000, 4.000], mean action: 1.792 [0.000, 3.000], mean observation: 39.437 [0.000, 142.000], loss: 0.029145, mean_squared_error: 31.345172, mean_q: 6.143203, mean_eps: 0.119264
      98026/500000: episode: 1358, duration: 11.916s, episode steps: 118, steps per second: 10, episode reward: 14.000, mean reward: 0.119 [0.000, 4.000], mean action: 2.051 [0.000, 3.000], mean observation: 39.289 [0.000, 142.000], loss: 0.030525, mean_squared_error: 30.888820, mean_q: 6.081656, mean_eps: 0.118301
      98121/500000: episode: 1359, duration: 9.505s, episode steps: 95, steps per second: 10, episode reward: 13.000, mean reward: 0.137 [0.000, 4.000], mean action: 1.768 [0.000, 3.000], mean observation: 39.416 [0.000, 142.000], loss: 0.028739, mean_squared_error: 31.910747, mean_q: 6.197497, mean_eps: 0.117343
      98199/500000: episode: 1360, duration: 7.894s, episode steps: 78, steps per second: 10, episode reward: 5.000, mean reward: 0.064 [0.000, 1.000], mean action: 1.590 [0.000, 3.000], mean observation: 39.590 [0.000, 142.000], loss: 0.030962, mean_squared_error: 31.646903, mean_q: 6.140278, mean_eps: 0.116564
      98279/500000: episode: 1361, duration: 8.131s, episode steps: 80, steps per second: 10, episode reward: 5.000, mean reward: 0.062 [0.000, 1.000], mean action: 1.600 [0.000, 3.000], mean observation: 39.591 [0.000, 142.000], loss: 0.036661, mean_squared_error: 32.434391, mean_q: 6.269695, mean_eps: 0.115853
      98385/500000: episode: 1362, duration: 10.763s, episode steps: 106, steps per second: 10, episode reward: 7.000, mean reward: 0.066 [0.000, 1.000], mean action: 1.575 [0.000, 3.000], mean observation: 39.390 [0.000, 142.000], loss: 0.027651, mean_squared_error: 31.947573, mean_q: 6.197611, mean_eps: 0.115016
      98482/500000: episode: 1363, duration: 9.787s, episode steps: 97, steps per second: 10, episode reward: 13.000, mean reward: 0.134 [0.000, 4.000], mean action: 1.629 [0.000, 3.000], mean observation: 39.447 [0.000, 142.000], loss: 0.032696, mean_squared_error: 32.207774, mean_q: 6.258774, mean_eps: 0.114103
      98602/500000: episode: 1364, duration: 12.206s, episode steps: 120, steps per second: 10, episode reward: 8.000, mean reward: 0.067 [0.000, 1.000], mean action: 2.025 [0.000, 3.000], mean observation: 39.294 [0.000, 142.000], loss: 0.030869, mean_squared_error: 32.008359, mean_q: 6.179322, mean_eps: 0.113126
      98687/500000: episode: 1365, duration: 8.611s, episode steps: 85, steps per second: 10, episode reward: 9.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.894 [0.000, 3.000], mean observation: 39.520 [0.000, 142.000], loss: 0.027585, mean_squared_error: 31.385419, mean_q: 6.124500, mean_eps: 0.112204
      98785/500000: episode: 1366, duration: 9.942s, episode steps: 98, steps per second: 10, episode reward: 10.000, mean reward: 0.102 [0.000, 4.000], mean action: 1.816 [0.000, 3.000], mean observation: 39.383 [0.000, 142.000], loss: 0.027373, mean_squared_error: 32.677929, mean_q: 6.282391, mean_eps: 0.111380
      98891/500000: episode: 1367, duration: 10.786s, episode steps: 106, steps per second: 10, episode reward: 14.000, mean reward: 0.132 [0.000, 4.000], mean action: 1.792 [0.000, 3.000], mean observation: 39.326 [0.000, 142.000], loss: 0.028836, mean_squared_error: 31.402265, mean_q: 6.135036, mean_eps: 0.110462
      99006/500000: episode: 1368, duration: 11.702s, episode steps: 115, steps per second: 10, episode reward: 11.000, mean reward: 0.096 [0.000, 4.000], mean action: 1.730 [0.000, 3.000], mean observation: 39.311 [0.000, 142.000], loss: 0.028842, mean_squared_error: 32.268523, mean_q: 6.252424, mean_eps: 0.109468
      99090/500000: episode: 1369, duration: 8.542s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.679 [0.000, 3.000], mean observation: 39.523 [0.000, 142.000], loss: 0.028907, mean_squared_error: 31.900966, mean_q: 6.185906, mean_eps: 0.108572
      99192/500000: episode: 1370, duration: 10.365s, episode steps: 102, steps per second: 10, episode reward: 14.000, mean reward: 0.137 [0.000, 4.000], mean action: 1.559 [0.000, 3.000], mean observation: 39.332 [0.000, 142.000], loss: 0.028146, mean_squared_error: 32.246745, mean_q: 6.270967, mean_eps: 0.107735
      99275/500000: episode: 1371, duration: 8.445s, episode steps: 83, steps per second: 10, episode reward: 12.000, mean reward: 0.145 [0.000, 4.000], mean action: 1.687 [0.000, 3.000], mean observation: 39.513 [0.000, 142.000], loss: 0.034852, mean_squared_error: 32.590558, mean_q: 6.291439, mean_eps: 0.106903
      99369/500000: episode: 1372, duration: 9.568s, episode steps: 94, steps per second: 10, episode reward: 13.000, mean reward: 0.138 [0.000, 4.000], mean action: 1.745 [0.000, 3.000], mean observation: 39.424 [0.000, 142.000], loss: 0.025090, mean_squared_error: 31.868699, mean_q: 6.215902, mean_eps: 0.106106
      99440/500000: episode: 1373, duration: 7.195s, episode steps: 71, steps per second: 10, episode reward: 4.000, mean reward: 0.056 [0.000, 1.000], mean action: 2.155 [0.000, 3.000], mean observation: 39.672 [0.000, 142.000], loss: 0.034307, mean_squared_error: 31.744117, mean_q: 6.180421, mean_eps: 0.105364
      99534/500000: episode: 1374, duration: 9.564s, episode steps: 94, steps per second: 10, episode reward: 13.000, mean reward: 0.138 [0.000, 4.000], mean action: 1.468 [0.000, 3.000], mean observation: 39.428 [0.000, 142.000], loss: 0.031380, mean_squared_error: 32.111669, mean_q: 6.236137, mean_eps: 0.104621
      99611/500000: episode: 1375, duration: 7.790s, episode steps: 77, steps per second: 10, episode reward: 8.000, mean reward: 0.104 [0.000, 4.000], mean action: 1.623 [0.000, 3.000], mean observation: 39.579 [0.000, 142.000], loss: 0.026582, mean_squared_error: 32.409027, mean_q: 6.257531, mean_eps: 0.103852
      99701/500000: episode: 1376, duration: 9.134s, episode steps: 90, steps per second: 10, episode reward: 13.000, mean reward: 0.144 [0.000, 4.000], mean action: 1.900 [0.000, 3.000], mean observation: 39.447 [0.000, 142.000], loss: 0.031303, mean_squared_error: 31.856154, mean_q: 6.175366, mean_eps: 0.103100
      99793/500000: episode: 1377, duration: 9.364s, episode steps: 92, steps per second: 10, episode reward: 9.000, mean reward: 0.098 [0.000, 4.000], mean action: 1.750 [0.000, 3.000], mean observation: 39.523 [0.000, 142.000], loss: 0.029230, mean_squared_error: 32.500729, mean_q: 6.269437, mean_eps: 0.102281
      99871/500000: episode: 1378, duration: 7.957s, episode steps: 78, steps per second: 10, episode reward: 8.000, mean reward: 0.103 [0.000, 4.000], mean action: 1.910 [0.000, 3.000], mean observation: 39.555 [0.000, 142.000], loss: 0.032038, mean_squared_error: 32.045452, mean_q: 6.251863, mean_eps: 0.101516
      99966/500000: episode: 1379, duration: 9.605s, episode steps: 95, steps per second: 10, episode reward: 13.000, mean reward: 0.137 [0.000, 4.000], mean action: 1.863 [0.000, 3.000], mean observation: 39.435 [0.000, 142.000], loss: 0.032570, mean_squared_error: 32.213077, mean_q: 6.227903, mean_eps: 0.100738
     100057/500000: episode: 1380, duration: 9.293s, episode steps: 91, steps per second: 10, episode reward: 13.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.879 [0.000, 3.000], mean observation: 39.417 [0.000, 142.000], loss: 0.031718, mean_squared_error: 31.927115, mean_q: 6.224809, mean_eps: 0.100059
     100154/500000: episode: 1381, duration: 9.844s, episode steps: 97, steps per second: 10, episode reward: 13.000, mean reward: 0.134 [0.000, 4.000], mean action: 2.000 [0.000, 3.000], mean observation: 39.410 [0.000, 142.000], loss: 0.028632, mean_squared_error: 31.639396, mean_q: 6.162611, mean_eps: 0.100000
     100236/500000: episode: 1382, duration: 8.317s, episode steps: 82, steps per second: 10, episode reward: 12.000, mean reward: 0.146 [0.000, 4.000], mean action: 1.634 [0.000, 3.000], mean observation: 39.493 [0.000, 142.000], loss: 0.028210, mean_squared_error: 32.374571, mean_q: 6.268237, mean_eps: 0.100000
     100326/500000: episode: 1383, duration: 9.103s, episode steps: 90, steps per second: 10, episode reward: 13.000, mean reward: 0.144 [0.000, 4.000], mean action: 1.944 [0.000, 3.000], mean observation: 39.426 [0.000, 142.000], loss: 0.030084, mean_squared_error: 32.497831, mean_q: 6.251474, mean_eps: 0.100000
     100410/500000: episode: 1384, duration: 8.544s, episode steps: 84, steps per second: 10, episode reward: 9.000, mean reward: 0.107 [0.000, 4.000], mean action: 1.798 [0.000, 3.000], mean observation: 39.503 [0.000, 142.000], loss: 0.030590, mean_squared_error: 32.844728, mean_q: 6.289803, mean_eps: 0.100000
     100524/500000: episode: 1385, duration: 11.587s, episode steps: 114, steps per second: 10, episode reward: 11.000, mean reward: 0.096 [0.000, 4.000], mean action: 1.658 [0.000, 3.000], mean observation: 39.267 [0.000, 142.000], loss: 0.029037, mean_squared_error: 31.999110, mean_q: 6.186816, mean_eps: 0.100000
     100594/500000: episode: 1386, duration: 7.134s, episode steps: 70, steps per second: 10, episode reward: 11.000, mean reward: 0.157 [0.000, 4.000], mean action: 1.914 [0.000, 3.000], mean observation: 39.554 [0.000, 142.000], loss: 0.027382, mean_squared_error: 31.773964, mean_q: 6.170306, mean_eps: 0.100000
     100691/500000: episode: 1387, duration: 9.779s, episode steps: 97, steps per second: 10, episode reward: 13.000, mean reward: 0.134 [0.000, 4.000], mean action: 1.649 [0.000, 3.000], mean observation: 39.433 [0.000, 142.000], loss: 0.028501, mean_squared_error: 32.434384, mean_q: 6.245947, mean_eps: 0.100000
     100787/500000: episode: 1388, duration: 9.667s, episode steps: 96, steps per second: 10, episode reward: 13.000, mean reward: 0.135 [0.000, 4.000], mean action: 1.625 [0.000, 3.000], mean observation: 39.432 [0.000, 142.000], loss: 0.034511, mean_squared_error: 32.123202, mean_q: 6.213338, mean_eps: 0.100000
     100873/500000: episode: 1389, duration: 8.684s, episode steps: 86, steps per second: 10, episode reward: 12.000, mean reward: 0.140 [0.000, 4.000], mean action: 1.663 [0.000, 3.000], mean observation: 39.501 [0.000, 142.000], loss: 0.030697, mean_squared_error: 31.722315, mean_q: 6.174478, mean_eps: 0.100000
     100956/500000: episode: 1390, duration: 8.377s, episode steps: 83, steps per second: 10, episode reward: 12.000, mean reward: 0.145 [0.000, 4.000], mean action: 1.602 [0.000, 3.000], mean observation: 39.512 [0.000, 142.000], loss: 0.031104, mean_squared_error: 32.582407, mean_q: 6.256197, mean_eps: 0.100000
     101071/500000: episode: 1391, duration: 11.718s, episode steps: 115, steps per second: 10, episode reward: 11.000, mean reward: 0.096 [0.000, 4.000], mean action: 2.061 [0.000, 3.000], mean observation: 39.264 [0.000, 142.000], loss: 0.027454, mean_squared_error: 32.249377, mean_q: 6.238426, mean_eps: 0.100000
     101156/500000: episode: 1392, duration: 8.625s, episode steps: 85, steps per second: 10, episode reward: 9.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.506 [0.000, 3.000], mean observation: 39.458 [0.000, 142.000], loss: 0.029789, mean_squared_error: 32.242835, mean_q: 6.212963, mean_eps: 0.100000
     101318/500000: episode: 1393, duration: 16.310s, episode steps: 162, steps per second: 10, episode reward: 13.000, mean reward: 0.080 [0.000, 4.000], mean action: 2.056 [0.000, 3.000], mean observation: 39.072 [0.000, 142.000], loss: 0.030716, mean_squared_error: 32.461875, mean_q: 6.257740, mean_eps: 0.100000
     101420/500000: episode: 1394, duration: 10.336s, episode steps: 102, steps per second: 10, episode reward: 10.000, mean reward: 0.098 [0.000, 4.000], mean action: 1.608 [0.000, 3.000], mean observation: 39.396 [0.000, 142.000], loss: 0.027459, mean_squared_error: 32.358007, mean_q: 6.244473, mean_eps: 0.100000
     101487/500000: episode: 1395, duration: 6.864s, episode steps: 67, steps per second: 10, episode reward: 4.000, mean reward: 0.060 [0.000, 1.000], mean action: 2.030 [0.000, 3.000], mean observation: 39.602 [0.000, 142.000], loss: 0.035630, mean_squared_error: 33.346304, mean_q: 6.325141, mean_eps: 0.100000
     101586/500000: episode: 1396, duration: 9.978s, episode steps: 99, steps per second: 10, episode reward: 10.000, mean reward: 0.101 [0.000, 4.000], mean action: 1.909 [0.000, 3.000], mean observation: 39.367 [0.000, 142.000], loss: 0.025879, mean_squared_error: 32.959446, mean_q: 6.307710, mean_eps: 0.100000
     101694/500000: episode: 1397, duration: 10.893s, episode steps: 108, steps per second: 10, episode reward: 11.000, mean reward: 0.102 [0.000, 4.000], mean action: 1.815 [0.000, 3.000], mean observation: 39.222 [0.000, 142.000], loss: 0.032342, mean_squared_error: 32.595009, mean_q: 6.268993, mean_eps: 0.100000
     101788/500000: episode: 1398, duration: 9.495s, episode steps: 94, steps per second: 10, episode reward: 13.000, mean reward: 0.138 [0.000, 4.000], mean action: 1.713 [0.000, 3.000], mean observation: 39.432 [0.000, 142.000], loss: 0.028311, mean_squared_error: 32.713500, mean_q: 6.295837, mean_eps: 0.100000
     101867/500000: episode: 1399, duration: 8.043s, episode steps: 79, steps per second: 10, episode reward: 9.000, mean reward: 0.114 [0.000, 4.000], mean action: 1.722 [0.000, 3.000], mean observation: 39.498 [0.000, 142.000], loss: 0.025228, mean_squared_error: 32.879184, mean_q: 6.300917, mean_eps: 0.100000
     102077/500000: episode: 1400, duration: 21.216s, episode steps: 210, steps per second: 10, episode reward: 12.000, mean reward: 0.057 [0.000, 4.000], mean action: 1.271 [0.000, 3.000], mean observation: 39.164 [0.000, 142.000], loss: 0.029668, mean_squared_error: 33.044815, mean_q: 6.308369, mean_eps: 0.100000
     102180/500000: episode: 1401, duration: 10.406s, episode steps: 103, steps per second: 10, episode reward: 10.000, mean reward: 0.097 [0.000, 4.000], mean action: 1.767 [0.000, 3.000], mean observation: 39.403 [0.000, 142.000], loss: 0.028103, mean_squared_error: 32.678241, mean_q: 6.259726, mean_eps: 0.100000
     102307/500000: episode: 1402, duration: 12.847s, episode steps: 127, steps per second: 10, episode reward: 13.000, mean reward: 0.102 [0.000, 4.000], mean action: 1.614 [0.000, 3.000], mean observation: 39.378 [0.000, 142.000], loss: 0.029351, mean_squared_error: 33.425858, mean_q: 6.356040, mean_eps: 0.100000
     102433/500000: episode: 1403, duration: 12.777s, episode steps: 126, steps per second: 10, episode reward: 12.000, mean reward: 0.095 [0.000, 4.000], mean action: 1.683 [0.000, 3.000], mean observation: 39.265 [0.000, 142.000], loss: 0.030855, mean_squared_error: 32.889901, mean_q: 6.283435, mean_eps: 0.100000
     102632/500000: episode: 1404, duration: 20.124s, episode steps: 199, steps per second: 10, episode reward: 15.000, mean reward: 0.075 [0.000, 4.000], mean action: 2.025 [0.000, 3.000], mean observation: 39.145 [0.000, 142.000], loss: 0.031131, mean_squared_error: 32.861037, mean_q: 6.284704, mean_eps: 0.100000
     102719/500000: episode: 1405, duration: 8.833s, episode steps: 87, steps per second: 10, episode reward: 9.000, mean reward: 0.103 [0.000, 4.000], mean action: 1.816 [0.000, 3.000], mean observation: 39.508 [0.000, 142.000], loss: 0.029602, mean_squared_error: 32.854535, mean_q: 6.274666, mean_eps: 0.100000
     102846/500000: episode: 1406, duration: 12.810s, episode steps: 127, steps per second: 10, episode reward: 16.000, mean reward: 0.126 [0.000, 4.000], mean action: 1.677 [0.000, 3.000], mean observation: 39.112 [0.000, 142.000], loss: 0.030934, mean_squared_error: 33.252413, mean_q: 6.340366, mean_eps: 0.100000
     102976/500000: episode: 1407, duration: 13.161s, episode steps: 130, steps per second: 10, episode reward: 8.000, mean reward: 0.062 [0.000, 1.000], mean action: 1.977 [0.000, 3.000], mean observation: 39.199 [0.000, 142.000], loss: 0.033152, mean_squared_error: 33.094007, mean_q: 6.308278, mean_eps: 0.100000
     103071/500000: episode: 1408, duration: 9.612s, episode steps: 95, steps per second: 10, episode reward: 13.000, mean reward: 0.137 [0.000, 4.000], mean action: 1.726 [0.000, 3.000], mean observation: 39.450 [0.000, 142.000], loss: 0.028390, mean_squared_error: 33.281948, mean_q: 6.341498, mean_eps: 0.100000
     103179/500000: episode: 1409, duration: 10.899s, episode steps: 108, steps per second: 10, episode reward: 14.000, mean reward: 0.130 [0.000, 4.000], mean action: 1.741 [0.000, 3.000], mean observation: 39.310 [0.000, 142.000], loss: 0.030608, mean_squared_error: 33.001639, mean_q: 6.308685, mean_eps: 0.100000
     103306/500000: episode: 1410, duration: 12.855s, episode steps: 127, steps per second: 10, episode reward: 14.000, mean reward: 0.110 [0.000, 4.000], mean action: 2.150 [0.000, 3.000], mean observation: 39.193 [0.000, 142.000], loss: 0.027780, mean_squared_error: 33.903296, mean_q: 6.413689, mean_eps: 0.100000
     103390/500000: episode: 1411, duration: 8.540s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.583 [0.000, 3.000], mean observation: 39.526 [0.000, 142.000], loss: 0.028965, mean_squared_error: 33.164420, mean_q: 6.316500, mean_eps: 0.100000
     103499/500000: episode: 1412, duration: 11.058s, episode steps: 109, steps per second: 10, episode reward: 10.000, mean reward: 0.092 [0.000, 4.000], mean action: 1.633 [0.000, 3.000], mean observation: 39.438 [0.000, 142.000], loss: 0.032335, mean_squared_error: 33.246237, mean_q: 6.342034, mean_eps: 0.100000
     103604/500000: episode: 1413, duration: 10.606s, episode steps: 105, steps per second: 10, episode reward: 11.000, mean reward: 0.105 [0.000, 4.000], mean action: 1.810 [0.000, 3.000], mean observation: 39.388 [0.000, 142.000], loss: 0.026740, mean_squared_error: 33.403475, mean_q: 6.344293, mean_eps: 0.100000
     103678/500000: episode: 1414, duration: 7.500s, episode steps: 74, steps per second: 10, episode reward: 11.000, mean reward: 0.149 [0.000, 4.000], mean action: 2.054 [0.000, 3.000], mean observation: 39.558 [0.000, 142.000], loss: 0.031866, mean_squared_error: 32.999926, mean_q: 6.277209, mean_eps: 0.100000
     103768/500000: episode: 1415, duration: 9.130s, episode steps: 90, steps per second: 10, episode reward: 13.000, mean reward: 0.144 [0.000, 4.000], mean action: 1.678 [0.000, 3.000], mean observation: 39.436 [0.000, 142.000], loss: 0.027464, mean_squared_error: 33.163760, mean_q: 6.326587, mean_eps: 0.100000
     103852/500000: episode: 1416, duration: 8.562s, episode steps: 84, steps per second: 10, episode reward: 9.000, mean reward: 0.107 [0.000, 4.000], mean action: 1.512 [0.000, 3.000], mean observation: 39.508 [0.000, 142.000], loss: 0.029118, mean_squared_error: 33.435523, mean_q: 6.315540, mean_eps: 0.100000
     103943/500000: episode: 1417, duration: 9.223s, episode steps: 91, steps per second: 10, episode reward: 9.000, mean reward: 0.099 [0.000, 4.000], mean action: 1.560 [0.000, 3.000], mean observation: 39.484 [0.000, 142.000], loss: 0.027676, mean_squared_error: 33.835350, mean_q: 6.357102, mean_eps: 0.100000
     104041/500000: episode: 1418, duration: 10.037s, episode steps: 98, steps per second: 10, episode reward: 13.000, mean reward: 0.133 [0.000, 4.000], mean action: 1.796 [0.000, 3.000], mean observation: 39.410 [0.000, 142.000], loss: 0.028149, mean_squared_error: 33.768337, mean_q: 6.398629, mean_eps: 0.100000
     104150/500000: episode: 1419, duration: 11.063s, episode steps: 109, steps per second: 10, episode reward: 12.000, mean reward: 0.110 [0.000, 4.000], mean action: 1.706 [0.000, 3.000], mean observation: 39.228 [0.000, 142.000], loss: 0.028189, mean_squared_error: 33.622506, mean_q: 6.360508, mean_eps: 0.100000
     104247/500000: episode: 1420, duration: 9.797s, episode steps: 97, steps per second: 10, episode reward: 13.000, mean reward: 0.134 [0.000, 4.000], mean action: 1.557 [0.000, 3.000], mean observation: 39.433 [0.000, 142.000], loss: 0.029637, mean_squared_error: 33.621693, mean_q: 6.372989, mean_eps: 0.100000
     104369/500000: episode: 1421, duration: 12.365s, episode steps: 122, steps per second: 10, episode reward: 11.000, mean reward: 0.090 [0.000, 4.000], mean action: 1.754 [0.000, 3.000], mean observation: 39.321 [0.000, 142.000], loss: 0.029043, mean_squared_error: 33.331123, mean_q: 6.334546, mean_eps: 0.100000
     104452/500000: episode: 1422, duration: 8.362s, episode steps: 83, steps per second: 10, episode reward: 12.000, mean reward: 0.145 [0.000, 4.000], mean action: 1.639 [0.000, 3.000], mean observation: 39.508 [0.000, 142.000], loss: 0.029739, mean_squared_error: 33.203812, mean_q: 6.277165, mean_eps: 0.100000
     104573/500000: episode: 1423, duration: 12.224s, episode steps: 121, steps per second: 10, episode reward: 15.000, mean reward: 0.124 [0.000, 4.000], mean action: 1.529 [0.000, 3.000], mean observation: 39.175 [0.000, 142.000], loss: 0.029125, mean_squared_error: 33.692074, mean_q: 6.342553, mean_eps: 0.100000
     104679/500000: episode: 1424, duration: 10.749s, episode steps: 106, steps per second: 10, episode reward: 11.000, mean reward: 0.104 [0.000, 4.000], mean action: 1.736 [0.000, 3.000], mean observation: 39.271 [0.000, 142.000], loss: 0.028521, mean_squared_error: 33.664145, mean_q: 6.345519, mean_eps: 0.100000
     104776/500000: episode: 1425, duration: 9.872s, episode steps: 97, steps per second: 10, episode reward: 13.000, mean reward: 0.134 [0.000, 4.000], mean action: 1.701 [0.000, 3.000], mean observation: 39.418 [0.000, 142.000], loss: 0.028489, mean_squared_error: 33.324955, mean_q: 6.302992, mean_eps: 0.100000
     104889/500000: episode: 1426, duration: 11.452s, episode steps: 113, steps per second: 10, episode reward: 14.000, mean reward: 0.124 [0.000, 4.000], mean action: 1.894 [0.000, 3.000], mean observation: 39.283 [0.000, 142.000], loss: 0.032620, mean_squared_error: 33.893304, mean_q: 6.362824, mean_eps: 0.100000
     104996/500000: episode: 1427, duration: 10.824s, episode steps: 107, steps per second: 10, episode reward: 14.000, mean reward: 0.131 [0.000, 4.000], mean action: 1.729 [0.000, 3.000], mean observation: 39.301 [0.000, 142.000], loss: 0.025825, mean_squared_error: 34.317225, mean_q: 6.415705, mean_eps: 0.100000
     105079/500000: episode: 1428, duration: 8.480s, episode steps: 83, steps per second: 10, episode reward: 12.000, mean reward: 0.145 [0.000, 4.000], mean action: 1.988 [0.000, 3.000], mean observation: 39.512 [0.000, 142.000], loss: 0.028854, mean_squared_error: 34.415088, mean_q: 6.455201, mean_eps: 0.100000
     105168/500000: episode: 1429, duration: 9.057s, episode steps: 89, steps per second: 10, episode reward: 13.000, mean reward: 0.146 [0.000, 4.000], mean action: 1.865 [0.000, 3.000], mean observation: 39.443 [0.000, 142.000], loss: 0.029814, mean_squared_error: 34.391131, mean_q: 6.428712, mean_eps: 0.100000
     105253/500000: episode: 1430, duration: 8.600s, episode steps: 85, steps per second: 10, episode reward: 12.000, mean reward: 0.141 [0.000, 4.000], mean action: 1.871 [0.000, 3.000], mean observation: 39.513 [0.000, 142.000], loss: 0.032934, mean_squared_error: 33.721624, mean_q: 6.359172, mean_eps: 0.100000
     105337/500000: episode: 1431, duration: 8.537s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.702 [0.000, 3.000], mean observation: 39.508 [0.000, 142.000], loss: 0.034966, mean_squared_error: 34.246894, mean_q: 6.429115, mean_eps: 0.100000
     105437/500000: episode: 1432, duration: 10.075s, episode steps: 100, steps per second: 10, episode reward: 10.000, mean reward: 0.100 [0.000, 4.000], mean action: 1.580 [0.000, 3.000], mean observation: 39.426 [0.000, 142.000], loss: 0.028330, mean_squared_error: 33.423583, mean_q: 6.321124, mean_eps: 0.100000
     105597/500000: episode: 1433, duration: 16.163s, episode steps: 160, steps per second: 10, episode reward: 14.000, mean reward: 0.087 [0.000, 4.000], mean action: 1.469 [0.000, 3.000], mean observation: 38.945 [0.000, 142.000], loss: 0.030662, mean_squared_error: 34.061681, mean_q: 6.403363, mean_eps: 0.100000
     105690/500000: episode: 1434, duration: 9.370s, episode steps: 93, steps per second: 10, episode reward: 10.000, mean reward: 0.108 [0.000, 4.000], mean action: 1.720 [0.000, 3.000], mean observation: 39.450 [0.000, 142.000], loss: 0.026922, mean_squared_error: 33.631431, mean_q: 6.354308, mean_eps: 0.100000
     105772/500000: episode: 1435, duration: 8.308s, episode steps: 82, steps per second: 10, episode reward: 12.000, mean reward: 0.146 [0.000, 4.000], mean action: 1.951 [0.000, 3.000], mean observation: 39.512 [0.000, 142.000], loss: 0.028768, mean_squared_error: 34.508416, mean_q: 6.443042, mean_eps: 0.100000
     105886/500000: episode: 1436, duration: 11.450s, episode steps: 114, steps per second: 10, episode reward: 11.000, mean reward: 0.096 [0.000, 4.000], mean action: 1.868 [0.000, 3.000], mean observation: 39.314 [0.000, 142.000], loss: 0.029784, mean_squared_error: 33.915880, mean_q: 6.371049, mean_eps: 0.100000
     105980/500000: episode: 1437, duration: 9.508s, episode steps: 94, steps per second: 10, episode reward: 6.000, mean reward: 0.064 [0.000, 1.000], mean action: 1.713 [0.000, 3.000], mean observation: 39.472 [0.000, 142.000], loss: 0.028344, mean_squared_error: 33.084372, mean_q: 6.271866, mean_eps: 0.100000
     106078/500000: episode: 1438, duration: 10.000s, episode steps: 98, steps per second: 10, episode reward: 10.000, mean reward: 0.102 [0.000, 4.000], mean action: 1.765 [0.000, 3.000], mean observation: 39.427 [0.000, 142.000], loss: 0.030424, mean_squared_error: 34.285078, mean_q: 6.416156, mean_eps: 0.100000
     106177/500000: episode: 1439, duration: 9.928s, episode steps: 99, steps per second: 10, episode reward: 10.000, mean reward: 0.101 [0.000, 4.000], mean action: 2.051 [0.000, 3.000], mean observation: 39.395 [0.000, 142.000], loss: 0.031722, mean_squared_error: 33.456683, mean_q: 6.346330, mean_eps: 0.100000
     106292/500000: episode: 1440, duration: 11.594s, episode steps: 115, steps per second: 10, episode reward: 15.000, mean reward: 0.130 [0.000, 4.000], mean action: 1.739 [0.000, 3.000], mean observation: 39.209 [0.000, 142.000], loss: 0.026336, mean_squared_error: 33.351375, mean_q: 6.309424, mean_eps: 0.100000
     106381/500000: episode: 1441, duration: 9.023s, episode steps: 89, steps per second: 10, episode reward: 9.000, mean reward: 0.101 [0.000, 4.000], mean action: 1.674 [0.000, 3.000], mean observation: 39.458 [0.000, 142.000], loss: 0.032119, mean_squared_error: 33.716282, mean_q: 6.327580, mean_eps: 0.100000
     106464/500000: episode: 1442, duration: 8.369s, episode steps: 83, steps per second: 10, episode reward: 12.000, mean reward: 0.145 [0.000, 4.000], mean action: 1.843 [0.000, 3.000], mean observation: 39.503 [0.000, 142.000], loss: 0.030418, mean_squared_error: 34.102942, mean_q: 6.384123, mean_eps: 0.100000
     106553/500000: episode: 1443, duration: 9.083s, episode steps: 89, steps per second: 10, episode reward: 6.000, mean reward: 0.067 [0.000, 1.000], mean action: 1.775 [0.000, 3.000], mean observation: 39.506 [0.000, 142.000], loss: 0.030407, mean_squared_error: 34.113781, mean_q: 6.377905, mean_eps: 0.100000
     106701/500000: episode: 1444, duration: 15.031s, episode steps: 148, steps per second: 10, episode reward: 12.000, mean reward: 0.081 [0.000, 4.000], mean action: 1.230 [0.000, 3.000], mean observation: 38.924 [0.000, 142.000], loss: 0.028702, mean_squared_error: 33.581342, mean_q: 6.339235, mean_eps: 0.100000
     106803/500000: episode: 1445, duration: 10.329s, episode steps: 102, steps per second: 10, episode reward: 14.000, mean reward: 0.137 [0.000, 4.000], mean action: 1.843 [0.000, 3.000], mean observation: 39.290 [0.000, 142.000], loss: 0.024899, mean_squared_error: 33.286406, mean_q: 6.318483, mean_eps: 0.100000
     106888/500000: episode: 1446, duration: 8.616s, episode steps: 85, steps per second: 10, episode reward: 12.000, mean reward: 0.141 [0.000, 4.000], mean action: 1.894 [0.000, 3.000], mean observation: 39.509 [0.000, 142.000], loss: 0.031841, mean_squared_error: 33.776970, mean_q: 6.373978, mean_eps: 0.100000
     107003/500000: episode: 1447, duration: 11.669s, episode steps: 115, steps per second: 10, episode reward: 11.000, mean reward: 0.096 [0.000, 4.000], mean action: 1.713 [0.000, 3.000], mean observation: 39.335 [0.000, 142.000], loss: 0.028763, mean_squared_error: 33.081773, mean_q: 6.280377, mean_eps: 0.100000
     107088/500000: episode: 1448, duration: 8.696s, episode steps: 85, steps per second: 10, episode reward: 13.000, mean reward: 0.153 [0.000, 4.000], mean action: 1.929 [0.000, 3.000], mean observation: 39.436 [0.000, 142.000], loss: 0.027511, mean_squared_error: 33.231103, mean_q: 6.327187, mean_eps: 0.100000
     107181/500000: episode: 1449, duration: 9.419s, episode steps: 93, steps per second: 10, episode reward: 10.000, mean reward: 0.108 [0.000, 4.000], mean action: 1.914 [0.000, 3.000], mean observation: 39.400 [0.000, 142.000], loss: 0.029654, mean_squared_error: 32.959888, mean_q: 6.279882, mean_eps: 0.100000
     107272/500000: episode: 1450, duration: 9.203s, episode steps: 91, steps per second: 10, episode reward: 13.000, mean reward: 0.143 [0.000, 4.000], mean action: 2.121 [0.000, 3.000], mean observation: 39.399 [0.000, 142.000], loss: 0.032990, mean_squared_error: 32.782907, mean_q: 6.243631, mean_eps: 0.100000
     107362/500000: episode: 1451, duration: 9.123s, episode steps: 90, steps per second: 10, episode reward: 9.000, mean reward: 0.100 [0.000, 4.000], mean action: 1.667 [0.000, 3.000], mean observation: 39.473 [0.000, 142.000], loss: 0.025341, mean_squared_error: 33.524503, mean_q: 6.365409, mean_eps: 0.100000
     107446/500000: episode: 1452, duration: 8.579s, episode steps: 84, steps per second: 10, episode reward: 12.000, mean reward: 0.143 [0.000, 4.000], mean action: 1.893 [0.000, 3.000], mean observation: 39.499 [0.000, 142.000], loss: 0.030387, mean_squared_error: 33.590625, mean_q: 6.361642, mean_eps: 0.100000
     107532/500000: episode: 1453, duration: 8.685s, episode steps: 86, steps per second: 10, episode reward: 12.000, mean reward: 0.140 [0.000, 4.000], mean action: 1.860 [0.000, 3.000], mean observation: 39.522 [0.000, 142.000], loss: 0.030613, mean_squared_error: 32.986225, mean_q: 6.308584, mean_eps: 0.100000
     107646/500000: episode: 1454, duration: 11.500s, episode steps: 114, steps per second: 10, episode reward: 14.000, mean reward: 0.123 [0.000, 4.000], mean action: 2.000 [0.000, 3.000], mean observation: 39.231 [0.000, 142.000], loss: 0.027111, mean_squared_error: 32.892107, mean_q: 6.255888, mean_eps: 0.100000
     107767/500000: episode: 1455, duration: 12.200s, episode steps: 121, steps per second: 10, episode reward: 13.000, mean reward: 0.107 [0.000, 4.000], mean action: 1.545 [0.000, 3.000], mean observation: 39.143 [0.000, 142.000], loss: 0.026769, mean_squared_error: 33.042057, mean_q: 6.275564, mean_eps: 0.100000
     107866/500000: episode: 1456, duration: 10.051s, episode steps: 99, steps per second: 10, episode reward: 13.000, mean reward: 0.131 [0.000, 4.000], mean action: 1.949 [0.000, 3.000], mean observation: 39.401 [0.000, 142.000], loss: 0.025654, mean_squared_error: 33.699436, mean_q: 6.359880, mean_eps: 0.100000
     107959/500000: episode: 1457, duration: 9.396s, episode steps: 93, steps per second: 10, episode reward: 13.000, mean reward: 0.140 [0.000, 4.000], mean action: 1.591 [0.000, 3.000], mean observation: 39.423 [0.000, 142.000], loss: 0.033045, mean_squared_error: 33.145013, mean_q: 6.316204, mean_eps: 0.100000
     108060/500000: episode: 1458, duration: 10.260s, episode steps: 101, steps per second: 10, episode reward: 14.000, mean reward: 0.139 [0.000, 4.000], mean action: 1.663 [0.000, 3.000], mean observation: 39.319 [0.000, 142.000], loss: 0.029150, mean_squared_error: 33.670082, mean_q: 6.356591, mean_eps: 0.100000
     108161/500000: episode: 1459, duration: 10.285s, episode steps: 101, steps per second: 10, episode reward: 10.000, mean reward: 0.099 [0.000, 4.000], mean action: 1.505 [0.000, 3.000], mean observation: 39.392 [0.000, 142.000], loss: 0.027632, mean_squared_error: 32.939406, mean_q: 6.273490, mean_eps: 0.100000
     108251/500000: episode: 1460, duration: 9.175s, episode steps: 90, steps per second: 10, episode reward: 9.000, mean reward: 0.100 [0.000, 4.000], mean action: 1.733 [0.000, 3.000], mean observation: 39.541 [0.000, 142.000], loss: 0.027815, mean_squared_error: 33.348284, mean_q: 6.321946, mean_eps: 0.100000
     108387/500000: episode: 1461, duration: 13.803s, episode steps: 136, steps per second: 10, episode reward: 12.000, mean reward: 0.088 [0.000, 4.000], mean action: 1.676 [0.000, 3.000], mean observation: 39.145 [0.000, 142.000], loss: 0.027126, mean_squared_error: 32.843614, mean_q: 6.264420, mean_eps: 0.100000
     108529/500000: episode: 1462, duration: 14.327s, episode steps: 142, steps per second: 10, episode reward: 15.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.810 [0.000, 3.000], mean observation: 39.081 [0.000, 142.000], loss: 0.027393, mean_squared_error: 33.435077, mean_q: 6.332471, mean_eps: 0.100000
     108606/500000: episode: 1463, duration: 7.859s, episode steps: 77, steps per second: 10, episode reward: 8.000, mean reward: 0.104 [0.000, 4.000], mean action: 2.000 [0.000, 3.000], mean observation: 39.593 [0.000, 142.000], loss: 0.032359, mean_squared_error: 33.181227, mean_q: 6.292578, mean_eps: 0.100000
     108712/500000: episode: 1464, duration: 10.688s, episode steps: 106, steps per second: 10, episode reward: 14.000, mean reward: 0.132 [0.000, 4.000], mean action: 1.943 [0.000, 3.000], mean observation: 39.306 [0.000, 142.000], loss: 0.030452, mean_squared_error: 33.255835, mean_q: 6.319420, mean_eps: 0.100000
     108815/500000: episode: 1465, duration: 10.378s, episode steps: 103, steps per second: 10, episode reward: 10.000, mean reward: 0.097 [0.000, 4.000], mean action: 1.680 [0.000, 3.000], mean observation: 39.417 [0.000, 142.000], loss: 0.026415, mean_squared_error: 33.987104, mean_q: 6.423493, mean_eps: 0.100000
     108905/500000: episode: 1466, duration: 9.061s, episode steps: 90, steps per second: 10, episode reward: 9.000, mean reward: 0.100 [0.000, 4.000], mean action: 1.878 [0.000, 3.000], mean observation: 39.469 [0.000, 142.000], loss: 0.026628, mean_squared_error: 32.896752, mean_q: 6.278669, mean_eps: 0.100000
     109028/500000: episode: 1467, duration: 12.394s, episode steps: 123, steps per second: 10, episode reward: 13.000, mean reward: 0.106 [0.000, 4.000], mean action: 1.740 [0.000, 3.000], mean observation: 39.160 [0.000, 142.000], loss: 0.027496, mean_squared_error: 32.896169, mean_q: 6.269494, mean_eps: 0.100000
     109139/500000: episode: 1468, duration: 11.244s, episode steps: 111, steps per second: 10, episode reward: 11.000, mean reward: 0.099 [0.000, 4.000], mean action: 1.495 [0.000, 3.000], mean observation: 39.325 [0.000, 142.000], loss: 0.025899, mean_squared_error: 33.052763, mean_q: 6.298639, mean_eps: 0.100000
     109235/500000: episode: 1469, duration: 9.732s, episode steps: 96, steps per second: 10, episode reward: 13.000, mean reward: 0.135 [0.000, 4.000], mean action: 1.917 [0.000, 3.000], mean observation: 39.417 [0.000, 142.000], loss: 0.025891, mean_squared_error: 32.891442, mean_q: 6.262330, mean_eps: 0.100000
     109354/500000: episode: 1470, duration: 12.036s, episode steps: 119, steps per second: 10, episode reward: 14.000, mean reward: 0.118 [0.000, 4.000], mean action: 2.042 [0.000, 3.000], mean observation: 39.208 [0.000, 142.000], loss: 0.028734, mean_squared_error: 33.535043, mean_q: 6.349479, mean_eps: 0.100000
     109470/500000: episode: 1471, duration: 11.673s, episode steps: 116, steps per second: 10, episode reward: 12.000, mean reward: 0.103 [0.000, 4.000], mean action: 1.810 [0.000, 3.000], mean observation: 39.325 [0.000, 142.000], loss: 0.029473, mean_squared_error: 32.924660, mean_q: 6.272394, mean_eps: 0.100000
     109606/500000: episode: 1472, duration: 13.737s, episode steps: 136, steps per second: 10, episode reward: 14.000, mean reward: 0.103 [0.000, 4.000], mean action: 2.176 [0.000, 3.000], mean observation: 39.199 [0.000, 142.000], loss: 0.031039, mean_squared_error: 33.443730, mean_q: 6.336935, mean_eps: 0.100000
     109701/500000: episode: 1473, duration: 9.638s, episode steps: 95, steps per second: 10, episode reward: 13.000, mean reward: 0.137 [0.000, 4.000], mean action: 1.747 [0.000, 3.000], mean observation: 39.427 [0.000, 142.000], loss: 0.026098, mean_squared_error: 33.643567, mean_q: 6.351299, mean_eps: 0.100000
     109798/500000: episode: 1474, duration: 9.843s, episode steps: 97, steps per second: 10, episode reward: 13.000, mean reward: 0.134 [0.000, 4.000], mean action: 1.907 [0.000, 3.000], mean observation: 39.407 [0.000, 142.000], loss: 0.032308, mean_squared_error: 33.298407, mean_q: 6.324128, mean_eps: 0.100000
     109899/500000: episode: 1475, duration: 10.268s, episode steps: 101, steps per second: 10, episode reward: 13.000, mean reward: 0.129 [0.000, 4.000], mean action: 1.752 [0.000, 3.000], mean observation: 39.380 [0.000, 142.000], loss: 0.028576, mean_squared_error: 33.653685, mean_q: 6.351502, mean_eps: 0.100000
     109990/500000: episode: 1476, duration: 9.280s, episode steps: 91, steps per second: 10, episode reward: 9.000, mean reward: 0.099 [0.000, 4.000], mean action: 1.956 [0.000, 3.000], mean observation: 39.450 [0.000, 142.000], loss: 0.026850, mean_squared_error: 33.689953, mean_q: 6.364325, mean_eps: 0.100000
     110080/500000: episode: 1477, duration: 9.155s, episode steps: 90, steps per second: 10, episode reward: 13.000, mean reward: 0.144 [0.000, 4.000], mean action: 1.756 [0.000, 3.000], mean observation: 39.450 [0.000, 142.000], loss: 0.031132, mean_squared_error: 34.192820, mean_q: 6.420163, mean_eps: 0.100000
     110156/500000: episode: 1478, duration: 7.672s, episode steps: 76, steps per second: 10, episode reward: 8.000, mean reward: 0.105 [0.000, 4.000], mean action: 1.671 [0.000, 3.000], mean observation: 39.542 [0.000, 142.000], loss: 0.023101, mean_squared_error: 34.425974, mean_q: 6.447395, mean_eps: 0.100000
     110265/500000: episode: 1479, duration: 10.996s, episode steps: 109, steps per second: 10, episode reward: 14.000, mean reward: 0.128 [0.000, 4.000], mean action: 1.569 [0.000, 3.000], mean observation: 39.350 [0.000, 142.000], loss: 0.023746, mean_squared_error: 33.726337, mean_q: 6.344825, mean_eps: 0.100000
     110368/500000: episode: 1480, duration: 10.379s, episode steps: 103, steps per second: 10, episode reward: 14.000, mean reward: 0.136 [0.000, 4.000], mean action: 1.777 [0.000, 3.000], mean observation: 39.330 [0.000, 142.000], loss: 0.030484, mean_squared_error: 34.071810, mean_q: 6.408295, mean_eps: 0.100000
     110460/500000: episode: 1481, duration: 9.363s, episode steps: 92, steps per second: 10, episode reward: 13.000, mean reward: 0.141 [0.000, 4.000], mean action: 1.837 [0.000, 3.000], mean observation: 39.418 [0.000, 142.000], loss: 0.022902, mean_squared_error: 34.093946, mean_q: 6.388820, mean_eps: 0.100000


# Testing

Here, we run the actual DQN in a test environment. There are a couple of things to note:

The `test` method doesn't necessarily use the same policy as the `fit` method.  It defaults to `GreedyQPolicy` unless `test_policy` in the `DQNAgent` instantiation is set.  If the model is properly trained, it's action might not vary at all given a greedy policy (e.g., it chooses `LEFT` for every action).  This isn't an issue (although it doesn't make for a very interesting training session), however, it can appear as the the session starts for a couple frames, then freezes.  This is actually because there is a `FIRE` action that the model must first take before the game starts.  If it's only going left, then the game never starts.  I added the 'frame' and 'action' labels to the visualization to check for this.

The `Visualization` callback needs to be added just like during training if you want to view it, and, similar to before, `visualize` needs to be set to `False` unless the system is properly configured to handle it.

Also similar to training, `action_repetition` is set to 4.  This is appropriate since this is how it was trained.


```python
callbacks = []
callbacks = [Visualizer()]

test_hist = dqn.test(env, nb_episodes=1, action_repetition=4,
                     callbacks=callbacks, visualize=False)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-9-b4bd5ecf2528> in <module>()
          3 
          4 test_hist = dqn.test(env, nb_episodes=1, action_repetition=4,
    ----> 5                      callbacks=callbacks, visualize=False)
    

    ~/anaconda3/lib/python3.6/site-packages/rl/core.py in test(self, env, nb_episodes, action_repetition, callbacks, visualize, nb_max_episode_steps, nb_max_start_steps, start_step_policy, verbose)
        351                     if self.processor is not None:
        352                         observation, r, d, info = self.processor.process_step(observation, r, d, info)
    --> 353                     callbacks.on_action_end(action)
        354                     reward += r
        355                     for key, value in info.items():


    ~/anaconda3/lib/python3.6/site-packages/rl/callbacks.py in on_action_end(self, action, logs)
         99         for callback in self.callbacks:
        100             if callable(getattr(callback, 'on_action_end', None)):
    --> 101                 callback.on_action_end(action, logs=logs)
        102 
        103 


    <ipython-input-3-fcb944301d1b> in on_action_end(self, action, logs)
         26             plt.annotate('action: ' + str(action), xy=(10, 50), color='white')
         27 
    ---> 28         display.display(plt.gcf())
         29         display.clear_output(wait=True)


    ~/anaconda3/lib/python3.6/site-packages/IPython/core/display.py in display(include, exclude, metadata, transient, display_id, *objs, **kwargs)
        295             publish_display_data(data=obj, metadata=metadata, **kwargs)
        296         else:
    --> 297             format_dict, md_dict = format(obj, include=include, exclude=exclude)
        298             if not format_dict:
        299                 # nothing to display (e.g. _ipython_display_ took over)


    ~/anaconda3/lib/python3.6/site-packages/IPython/core/formatters.py in format(self, obj, include, exclude)
        178             md = None
        179             try:
    --> 180                 data = formatter(obj)
        181             except:
        182                 # FIXME: log the exception


    <decorator-gen-9> in __call__(self, obj)


    ~/anaconda3/lib/python3.6/site-packages/IPython/core/formatters.py in catch_format_error(method, self, *args, **kwargs)
        222     """show traceback on failed format call"""
        223     try:
    --> 224         r = method(self, *args, **kwargs)
        225     except NotImplementedError:
        226         # don't warn on NotImplementedErrors


    ~/anaconda3/lib/python3.6/site-packages/IPython/core/formatters.py in __call__(self, obj)
        339                 pass
        340             else:
    --> 341                 return printer(obj)
        342             # Finally look for special method names
        343             method = get_real_method(obj, self.print_method)


    ~/anaconda3/lib/python3.6/site-packages/IPython/core/pylabtools.py in <lambda>(fig)
        236 
        237     if 'png' in formats:
    --> 238         png_formatter.for_type(Figure, lambda fig: print_figure(fig, 'png', **kwargs))
        239     if 'retina' in formats or 'png2x' in formats:
        240         png_formatter.for_type(Figure, lambda fig: retina_figure(fig, **kwargs))


    ~/anaconda3/lib/python3.6/site-packages/IPython/core/pylabtools.py in print_figure(fig, fmt, bbox_inches, **kwargs)
        120 
        121     bytes_io = BytesIO()
    --> 122     fig.canvas.print_figure(bytes_io, **kw)
        123     data = bytes_io.getvalue()
        124     if fmt == 'svg':


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/backend_bases.py in print_figure(self, filename, dpi, facecolor, edgecolor, orientation, format, **kwargs)
       2198                     orientation=orientation,
       2199                     dryrun=True,
    -> 2200                     **kwargs)
       2201                 renderer = self.figure._cachedRenderer
       2202                 bbox_inches = self.figure.get_tightbbox(renderer)


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py in print_png(self, filename_or_obj, *args, **kwargs)
        543 
        544     def print_png(self, filename_or_obj, *args, **kwargs):
    --> 545         FigureCanvasAgg.draw(self)
        546         renderer = self.get_renderer()
        547         original_dpi = renderer.dpi


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py in draw(self)
        462 
        463         try:
    --> 464             self.figure.draw(self.renderer)
        465         finally:
        466             RendererAgg.lock.release()


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         61     def draw_wrapper(artist, renderer, *args, **kwargs):
         62         before(artist, renderer)
    ---> 63         draw(artist, renderer, *args, **kwargs)
         64         after(artist, renderer)
         65 


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/figure.py in draw(self, renderer)
       1142 
       1143             mimage._draw_list_compositing_images(
    -> 1144                 renderer, self, dsu, self.suppressComposite)
       1145 
       1146             renderer.close_group('figure')


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/image.py in _draw_list_compositing_images(renderer, parent, dsu, suppress_composite)
        137     if not_composite or not has_images:
        138         for zorder, a in dsu:
    --> 139             a.draw(renderer)
        140     else:
        141         # Composite any adjacent images together


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         61     def draw_wrapper(artist, renderer, *args, **kwargs):
         62         before(artist, renderer)
    ---> 63         draw(artist, renderer, *args, **kwargs)
         64         after(artist, renderer)
         65 


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_base.py in draw(self, renderer, inframe)
       2424             renderer.stop_rasterizing()
       2425 
    -> 2426         mimage._draw_list_compositing_images(renderer, self, dsu)
       2427 
       2428         renderer.close_group('axes')


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/image.py in _draw_list_compositing_images(renderer, parent, dsu, suppress_composite)
        137     if not_composite or not has_images:
        138         for zorder, a in dsu:
    --> 139             a.draw(renderer)
        140     else:
        141         # Composite any adjacent images together


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/artist.py in draw_wrapper(artist, renderer, *args, **kwargs)
         61     def draw_wrapper(artist, renderer, *args, **kwargs):
         62         before(artist, renderer)
    ---> 63         draw(artist, renderer, *args, **kwargs)
         64         after(artist, renderer)
         65 


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axis.py in draw(self, renderer, *args, **kwargs)
       1134         renderer.open_group(__name__)
       1135 
    -> 1136         ticks_to_draw = self._update_ticks(renderer)
       1137         ticklabelBoxes, ticklabelBoxes2 = self._get_tick_bboxes(ticks_to_draw,
       1138                                                                 renderer)


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axis.py in _update_ticks(self, renderer)
        967 
        968         interval = self.get_view_interval()
    --> 969         tick_tups = [t for t in self.iter_ticks()]
        970         if self._smart_bounds:
        971             # handle inverted limits


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axis.py in <listcomp>(.0)
        967 
        968         interval = self.get_view_interval()
    --> 969         tick_tups = [t for t in self.iter_ticks()]
        970         if self._smart_bounds:
        971             # handle inverted limits


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axis.py in iter_ticks(self)
        911         """
        912         majorLocs = self.major.locator()
    --> 913         majorTicks = self.get_major_ticks(len(majorLocs))
        914         self.major.formatter.set_locs(majorLocs)
        915         majorLabels = [self.major.formatter(val, i)


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axis.py in get_major_ticks(self, numticks)
       1322             # update the new tick label properties from the old
       1323             for i in range(numticks - len(self.majorTicks)):
    -> 1324                 tick = self._get_tick(major=True)
       1325                 self.majorTicks.append(tick)
       1326 


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axis.py in _get_tick(self, major)
       1727         else:
       1728             tick_kw = self._minor_tick_kw
    -> 1729         return XTick(self.axes, 0, '', major=major, **tick_kw)
       1730 
       1731     def _get_label(self):


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axis.py in __init__(self, axes, loc, label, size, width, color, tickdir, pad, labelsize, labelcolor, zorder, gridOn, tick1On, tick2On, label1On, label2On, major)
        148         self.apply_tickdir(tickdir)
        149 
    --> 150         self.tick1line = self._get_tick1line()
        151         self.tick2line = self._get_tick2line()
        152         self.gridline = self._get_gridline()


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/axis.py in _get_tick1line(self)
        418                           linestyle='None', marker=self._tickmarkers[0],
        419                           markersize=self._size,
    --> 420                           markeredgewidth=self._width, zorder=self._zorder)
        421         l.set_transform(self.axes.get_xaxis_transform(which='tick1'))
        422         self._set_artist_props(l)


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/lines.py in __init__(self, xdata, ydata, linewidth, linestyle, color, marker, markersize, markeredgewidth, markeredgecolor, markerfacecolor, markerfacecoloralt, fillstyle, antialiased, dash_capstyle, solid_capstyle, dash_joinstyle, solid_joinstyle, pickradius, drawstyle, markevery, **kwargs)
        403         self.set_color(color)
        404         self._marker = MarkerStyle()
    --> 405         self.set_marker(marker)
        406 
        407         self._markevery = None


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/lines.py in set_marker(self, marker)
       1169 
       1170         """
    -> 1171         self._marker.set_marker(marker)
       1172         self.stale = True
       1173 


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/markers.py in set_marker(self, marker)
        270 
        271         self._marker = marker
    --> 272         self._recache()
        273 
        274     def get_path(self):


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/markers.py in _recache(self)
        206         self._capstyle = 'butt'
        207         self._filled = True
    --> 208         self._marker_function()
        209 
        210     if six.PY3:


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/markers.py in _set_tickdown(self)
        720 
        721     def _set_tickdown(self):
    --> 722         self._transform = Affine2D().scale(1.0, -1.0)
        723         self._snap_threshold = 1.0
        724         self._filled = False


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/transforms.py in __init__(self, matrix, **kwargs)
       1816         If *matrix* is None, initialize with the identity transform.
       1817         """
    -> 1818         Affine2DBase.__init__(self, **kwargs)
       1819         if matrix is None:
       1820             matrix = np.identity(3)


    ~/anaconda3/lib/python3.6/site-packages/matplotlib/transforms.py in __init__(self, *args, **kwargs)
       1661 
       1662     def __init__(self, *args, **kwargs):
    -> 1663         Transform.__init__(self, *args, **kwargs)
       1664         self._inverted = None
       1665 


    KeyboardInterrupt: 



![png](DQN_files/DQN_18_1.png)



![png](DQN_files/DQN_18_2.png)


# Results

After we have trained everything, we can look at the training history. We then plot the `episode_reward` to see how well it performed over each episode.


```python
results = pd.read_pickle("sessions/" + train_name +"/hist.pickle")
plt.plot(results['episode_reward'])
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAgAElEQVR4XuydB1yVZf//P2wFBffCiVsBceEEN+7HVVaOyjK10jRNcyW4MjVHjzkqzVGpaVqOSgHFBS6UpbgXuFBRAQeb/+u6k//PeFDPOdc5h3PO/blfr17P4Ppe431/b3pzj+9lBR4kQAIkQAIkQAIkQAKqImClqtVysSRAAiRAAiRAAiRAAqAAMglIgARIgARIgARIQGUEKIAqO+FcLgmQAAmQAAmQAAlQAJkDJEACJEACJEACJKAyAhRAlZ1wLpcESIAESIAESIAEKIDMARIgARIgARIgARJQGQEKoMpOOJdLAiRAAiRAAiRAAhRA5gAJkAAJkAAJkAAJqIwABVBlJ5zLJQESIAESIAESIAEKIHOABEiABEiABEiABFRGgAKoshPO5ZIACZAACZAACZAABZA5QAIkQAIkQAIkQAIqI0ABVNkJ53JJgARIgARIgARIgALIHCABEiABEiABEiABlRGgAKrshHO5JEACJEACJEACJEABZA6QAAmQAAmQAAmQgMoIUABVdsK5XBIgARIgARIgARKgADIHSIAESIAESIAESEBlBCiAKjvhXC4JkAAJkAAJkAAJUACZAyRAAiRAAiRAAiSgMgIUQJWdcC6XBEiABEiABEiABCiAzAESIAESIAESIAESUBkBCqDKTjiXSwIkQAIkQAIkQAIUQOYACZAACZAACZAACaiMAAVQZSecyyUBEiABEiABEiABCiBzgARIgARIgARIgARURoACqLITzuWSAAmQAAmQAAmQAAWQOUACJEACJEACJEACKiNAAVTZCedySYAESIAESIAESIACyBwgARIgARIgARIgAZURoACq7IRzuSRAAiRAAiRAAiRAAWQOkAAJkAAJkAAJkIDKCFAAVXbCuVwSIAESIAESIAESoAAyB0iABEiABEiABEhAZQQogCo74VwuCZAACZAACZAACVAAmQMkQAIkQAIkQAIkoDICFECVnXAulwRIgARIgARIgAQogMwBEiABEiABEiABElAZAQqgyk44l0sCJEACJEACJEACFEDmAAmQAAmQAAmQAAmojAAFUGUnnMslARIgARIgARIgAQogc4AESIAESIAESIAEVEaAAqiyE87lkgAJkAAJkAAJkAAFkDlAAiRAAiRAAiRAAiojQAFU2QnnckmABEiABEiABEiAAsgcIAESIAESIAESIAGVEaAAquyEc7kkQAIkQAIkQAIkQAFkDpAACZAACZAACZCAyghQAFV2wrlcEiABEiABEiABEqAAMgdIgARIgARIgARIQGUEKIAqO+FcLgmQAAmQAAmQAAlQAJkDJEACJEACJEACJKAyAhRAlZ1wLpcESIAESIAESIAEKIDMARIgARIgARIgARJQGQEKoMpOOJdLAiRAAiRAAiRAAhRA5gAJkAAJkAAJkAAJqIwABVDuhAt+FQCkyHXDaBIgARIgARIgASMTKArgJoAcI49rEsNRAOVOgyuA63JdMJoESIAESIAESKCACFQEcKOAxi7QYSmAcvidASTFx8fD2Vn8Vx4kQAIkQAIkQAKmTiA5ORmVKlUS03QBkGzq8zXE/CiAclQVAUxKSqIAynFkNAmQAAmQAAkYjYAQQBcX4X4UQKNBt7CBKIAWdkK5HBIgARIgAcsnQAEEeAdQLs8pgHL8GE0CJEACJEACRidAAaQAyiYdBVCWIONJgARIgARIwMgEKIAUQNmUowDKEmQ8CZAACZAACRiZAAWQAiibchRAWYKMJwESIAESIAEjE6AAUgBlU44CKEuQ8SRAAiRAAiRgZAIUQAqgbMpRAGUJMp4ESIAESIAEjEyAAkgBlE05CqAsQcaTAAmQAAmQgJEJUAApgLIpRwGUJch4EiABEiABEjAyAQogBfAqgCr55N0yAB9rkI8UQA0gsQkJkAAJkAAJmBIBCiAFsDQAm+eS0h1AEIB2APZpkKwUQA0gsQkJkAAJkAAJmBIBCiAFMG8+LgbQA0BNADkaJCsFUANIbEICJEACJEACpkSAAkgBfD4f7QHcBLAQwJcvSFQHAOKf3KMogOtJSUlwdhYuyIMESMBSCGw9eV1ZSt9GFS1lSVwHCZDAMwIUQArg8xdDfwDrAVR+JoL5XSgBAPzz/oACyN8pJGBZBO6kpKLZl3uQkwMcmdQB5VwKWdYCuRoSUDkBCiAF8PlLYDeAdAA9X3Jd8A6gyn9pcPnqILDr1G2M+PmEstiF/RvwLqA6TjtXqSICFEAKYG66iy+BL4unPQC2aXEN8B1ALWCxKQmYC4E5f53BdwfErwSgX6OKWNC/gblMnfMkARLQgAAFkAKYmybi0e5wAJUAZGqQO7lNKIBawGJTEjAXAq+vCMPxqw+U6ZZ3KYSwie1hZWVlLtPnPEmABF5BgAJIARQpYg3gCoANACZqedVQALUExuYkYOoE0jOz4R6wG+I/c4+Qz9qiWiknU58650cCJKAhAQogBVCkih8A8f5fbQDnNcwd3gHUEhSbk4C5EIiMf4jeS0NR3NEONcsWxbEr9zGrtzsGNc+vZry5rIrzJAESeJ4ABZACKHtF8A6gLEHGk4CJEfjx0BXM2BmL9nXKwKtSMSwMOo9uHuWwbGBjE5spp0MCJKArAQogBVDX3OEdQFlyjCcBEyUwcv1J7Iy+hc/8aqFF9ZLot/ywcjfwxNROsLbme4Ameto4LRLQigAFkAKoVcLk05h3AGUJMp4ETIxAyzl7cDMpFes/aIamVUvAa3ogHqdnYeeo1nB3dTGx2XI6JEACuhCgAFIAdcmb52MogLIEGU8CJkTgdlIqms/ZA3GjLyagM5wcbDFk9TGEnLuLKd3q4gNfNxOaLadCAiSgKwEKIAVQ19zhI2BZcownARMk8FfMLXz0y0nUK++Mv0b7KDNcefAyZv15Bm1rl8aaId4mOGtOiQRIQFsCFEAKoLY5k7c97wDKEmQ8CZgQgZk7Y7Hq0BUMal4Zs3p7KDOLvZmMbv89CEd7G0RO84O9ragcxYMESMCcCVAAKYCy+UsBlCXIeBIwIQJ9loUiIu4hFr3RAH0aVlRmlp2dgyazg3H/cTo2j2ihvBfIgwRIwLwJUAApgLIZTAGUJch4EjARAmmZWfDwD0R6Vjb2j2+LKiX/r/Dzx+tP4s/oWxjTsSbGdKxlIjPmNEiABHQlQAGkAOqaO7lxFEBZgownARMhcOLaA/RbHoaSTvYIn9rxX1u//XL0Gqb8fgre1Upg0/AWJjJjToMESEBXAhRACqCuuUMBlCXHeBIwMQI/HLiM2X+dQce6ZbHynSb/mt3Ve4/R9ut9sLOxQpS/HxztbU1s9pwOCZCANgQogBRAbfIlv7a8AyhLkPEkYCIEPvz5BP4+dRufd6mDD9tW/9escnJy0HpuCG48fIq173mjTa3SJjJrToMESEAXAhRACqAuefN8DAVQliDjScAECAjBE/X/EpLT8Ouw5mjmVvJ/ZjV+cxQ2n7iO4b5umNStrgnMmlMgARLQlQAFkAKoa+7kxlEAZQkyngRMgIC4s9fqq72wtbZSCkAXtrf5n1n9EXEDY36NhLurM3aO+qdGIA8SIAHzJEABpADKZi4FUJYg40nABAhsj7qJTzZEwMPVBTtGtc53RgnJqWj25R5YWQERX3RCMUd7E5g5p0ACJKALAQogBVCXvHk+hgIoS5DxJGACBAK2n8aasKt4t2VVBPyn/gtn1HHhfly88wgrBjVCF/fyJjBzToEESEAXAhRACqAueUMBlKXGeBIwMQK9vj2EqOtJ+OZNL/Tycn3h7Py3ncLaw9cwuHkVzOztbmKr4HRIgAQ0JUABpABqmisvasc7gLIEGU8CBUwgNSML7v67kZmdg4MT2qFSCccXzmj36dsY/tMJuJV2wt5xbQt45hyeBEhAVwIUQAqgrrmTG0cBlCXIeBIoYALHrtxH/+8Oo3RRBxyb3OFfBaDzTi3paQYazghEdg5wZFIHlHMpVMCz5/AkQAK6EKAAUgB1yZvnYyiAsgQZTwIFTGDF/kv46u+z6FK/HFYMbvzK2fzn20OIvp6Ehf0boG+jf/YL5kECJGBeBCiAFEDZjKUAyhJkPAkUMIFh68IRGJuAyd3qYJjvvwtA5zc1IYtCGvs1qogF/RsU8Ow5PAmQgC4EKIAUQF3yhncAZakxngRMhIAoAN109h7ce5SG30a0QJOqJV45s4MX7mLwqmMo71IIYRPbv/SR8Ss7YwMSIIECIUABpADKJh7vAMoSZDwJFCCBuMQn8J0fouzxKwpAF7L73wLQeaf3ND0LDaYHIj0rG3vHtYFb6SIFuAIOTQIkoAsBCiAFUJe84R1AWWqMJwETIZC7u4dXpWL44+NWGs/qze8P48jl+0opGFEShgcJkIB5EaAAUgBlM5Z3AGUJMp4ECpDAtG2nsO7wNbzXqhqm9ayn8Uz+u+cCFgadRzePclg28NUfjmjcMRuSAAkYhQAFkAIom2gUQFmCjCeBAiTQY8lBnLqRjG8HNEQPzwoaz+TEtfvot/wwijva4cTUTrC2ttI4lg1JgAQKngAFkAIom4UUQFmCjCeBAiLwJD0THgGByMrOweFJ7VHepbDGM8nIyobX9EA8Ts/CzlGt4e7qonEsG5IACRQ8AQogBVA2CymAsgQZTwIFRODwpUS89cMR5Wvew5M6aD2L99Ycx96zdzQuH6P1AAwgARIwGAEKIAVQNrkogLIEGU8CBURgachFzN99Dt09ymPpwEZaz2LlwcuY9ecZtKlVGmvf89Y6ngEkQAIFR4ACSAGUzT4KoCxBxpNAAREYuvY4gs/cwdTudTHUx03rWcTeTEa3/x6Eo70NIqf5wd7WWus+GEACJFAwBCiAFEDZzKMAyhJkPAkUAAFRALrxrGDcf5yO3z9qiYaVi2s9i+zsHDSZ/U8fm0e0QFMNikhrPQgDSIAEDEKAAkgBlE0sCqAsQcaTQAEQuHLvMdp9vU+5a3cqoLPOd+8+Xn8Sf0bfwpiONTGmY60CWAmHJAES0IUABZACqEvePB9DAZQlyHgSKAACW05cx7jNUWhcpTi2fNhS5xmsPxqHyb/HwLtqCWwa0ULnfhhIAiRgXAIUQAqgbMZRAGUJMp4ECoCAkDYhbx/4VMOU7poXgM471WuJj9Fm/j5lK7kofz842tsWwGo4JAmQgLYEKIAUQG1zJm97CqAsQcaTQAEQ6LL4AM7eTsGKQY3Qxb28zjMQ7xK2nhuCGw+fKl8Ciy+CeZAACZg+AQogBVA2SymAsgQZTwJGJvAoLROeAbuRnQMcm9wBZZwLSc1g/OYobD5xHcN93TCpW12pvhhMAiRgHAIUQAqgbKZRAGUJMp4EjEwg9OI9DFx5FK7FCiN0Ynvp0f+IuIExv0bC3dUZO0f5SPfHDkiABAxPgAJIAZTNMgqgLEHGk4CRCfx3zwUsDDqPng0qYMlbDaVHv5OcCu8v98DKCoj4ohOKOdpL98kOSIAEDEuAAkgBlM0wCqAsQcaTgJEJvLv6GPadu4uAnvXwbqtqehm908L9uHDnEZYPbISuHrq/U6iXybATEiCBVxKgAFIAXQHMBdAVgCOAiwCGAAh/Zfb804ACqCEoNiMBUyAgijc3nBmEpKcZ2D6yFTwrFtPLtAK2n8aasKsY3LwKZvZ210uf7IQESMBwBCiA6hZAUfo/AkAIgOUA7gKoCeDSs380yTwKoCaU2IYETITAxTuP0HHhfhSys0ZMQGfY2ehn+7bdp29j+E8n4FbaCXvHtTWR1XIaJEACLyJAAVS3AH4FoBUAmbe2KYD8/UICZkRg0/F4TNgSrffCzeKOYsMZgcqXxUcmdUA5F7kvi80IKadKAmZJgAKobgGMBbAbQEUAbQDcALAMwA8vyWYHAOKf3KMogOtJSUlwdhYuyIMESMCUCUzcEo2Nx+Mxok11TOxaR69T7fXtIURdT8KC1xugX2Pxa4UHCZCAqRKgAKpbAFOfJeZCAJsBNAXwDYARANa+IGkDAPjn/RkF0FQvcc6LBP5NwG/RfpxPeITvBzeGX/1yesUzd9dZLN93CX0buWJhfy+99s3OSIAE9EuAAqhuAUx/9rHH8xuB/veZCL5oU0/eAdTvNcjeSMBoBMRjWq8ZgcjJAcKndkSpIs/fzJefxqEL9zBo1VGUdymEsIntYSXqwvAgARIwSQIUQHUL4DUAQQCGPpedHwKYCkB8HazJwXcANaHENiRgAgT2n7+Ld348hiolHbF/fDu9z+hpehYaTA9EelY29o5rA7fSRfQ+BjskARLQDwEKoLoFcD2ASnk+AlkEoBmA5+8KvizbKID6uRbZCwkYnMCioPP4Zs8F9GnoikVvGOYR7ZvfH8aRy/eVUjCiJAwPEiAB0yRAAVS3AIp3/sKevdO3CYD3sw9AhgH4RcOUpQBqCIrNSKCgCQxedRQHL9zDzF71MbhFVYNMZ8meC1gQdB5d3cth+aDGBhmDnZIACcgToACqWwBFBvUAMOdZ/b8rAMQHIS/7Cjhv1lEA5a9D9kACBicgCkCLx7MpaZnYOao13F1dDDLmiWsP0G95GIo52uHk1E6wtuZ7gAYBzU5JQJIABZACKJlC3AlEFiDjScAYBM7dTkHnxQfgaG+DaH8/2OqpAHTeuWdmZcNrRhAeGVg0jcGMY5CAJROgAFIAZfObdwBlCTKeBIxAYMOxOEzaGoMWbiWxYVhzg4743prj2Hv2DiZ3q4NhvtUNOhY7JwES0I0ABZACqFvm/F8UBVCWIONJwAgExm+OwuYT1/Fxu+oY31m/BaDzTn/lwcuY9ecZtKlVGmvfE68W8yABEjA1AhRACqBsTlIAZQkyngSMQKDDgn24dPcxVr3TBB3qljXoiGduJaPrNwdR2M4GUf5+sLfVz37DBp00OycBlRGgAFIAZVOeAihLkPEkYGACD5+kK+/liePkF51QwsneoCOKD06azg5G4uN0bBreAt7VShh0PHZOAiSgPQEKIAVQ+6z5dwQFUJYg40nAwARCzt7BkDXH4VbKCXs/a2vg0f7pfuT6k9gZfQtjOtbEmI61jDImByEBEtCcAAWQAqh5tuTfkgIoS5DxJGBgAgsCz2HJ3ovo16giFvRvYODR/ul+/dE4TP49Bt5VS2DTiBftLGmUqXAQEiCBfAhQACmAshcGBVCWIONJwMAEBq48gtCLiZjdxx0Dmxlnd45riY/RZv4+2NlYKe8BOtrbGniV7J4ESEAbAhRACqA2+ZJfWwqgLEHGk4ABCWRl58AzYDcep2dh1xgf1CknLlnDHzk5OWg9NwQ3Hj7FmiFN0bZ2GcMPyhFIgAQ0JkABpABqnCwvaEgBlCXIeBIwIIHYm8no9t+DKOJgq9yJszHizhwTfovCpvDrGObrhsnd6hpwleyaBEhAWwIUQAqgtjmTtz0FUJYg40nAgAR+PnINU/84hdY1SuHnoc0MONL/dr0t8gZGb4yEu6szdo7yMerYHIwESODlBCiAFEDZa4QCKEuQ8SRgQAJjf43E1ogb+KR9DYz1q23Akf636zvJqfD+cg+srICILzqhmKNhy88YdXEcjATMnAAFkAIom8IUQFmCjCcBAxJoOz8EVxOfFNh7eJ0W7seFO4+wfGAjdPUob8CVsmsSIAFtCFAAKYDa5Et+bSmAsgQZTwIGIpD4KA2NZwUrvUdN84OLo52BRnpxtwHbT2NN2FUMal4Zs3p7GH18DkgCJJA/AQogBVD22qAAyhJkPAkYiEBwbAKGrgtHjTJFEDy2jYFGeXm3gadvY9hPJ4xahLpAFspBScDMCFAAKYCyKUsBlCXIeBIwEIG5u85i+b5L6N+kIua9ZpwC0HmXkvQ0Aw1nBCI7Bzg8qT3KuxQ20GrZLQmQgDYEKIAUQG3yJb+2FEBZgownAQMReOO7wzh65T7m9vPAG00rG2iUV3fb69tDiLqehAWvN0C/xhVfHcAWJEACBidAAaQAyiYZBVCWIONJwAAEMrOy4REQiKcZWQj61Bc1yxY1wCiadZl7J7JvI1cs7O+lWRBbkQAJGJQABZACKJtgFEBZgownAQMQOHUjCT2WHIJzIVtETvODtRELQOddzqEL9zBo1VGUcy6kPAa2EnVheJAACRQoAQogBVA2ASmAsgQZTwIGILA27Cr8t5+Gb63SWPeetwFG0LzL1IwseE4PRHpmNvaMa4PqpYtoHsyWJEACBiFAAaQAyiYWBVCWIONJwAAERm+MwLbIm/i0Yy2M7ljTACNo1+Vb3x/B4cuJmNmrPga3qKpdMFuTAAnonQAFkAIom1QUQFmCjCcBAxDwmbcX8fef4qf3veFTs7QBRtCuyyV7LmBB0Hl0dS+H5YMaaxfM1iRAAnonQAGkAMomFQVQliDjSUDPBO6kpMJ79j9bsEX7+6FoIeMXgM67pBPXHqDf8jAUc7TDyamdCvSdRD3jZnckYJYEKIAUQNnEpQDKEmQ8CeiZwK5TtzHi5xOoU64odo3x1XPvunUnvkr2mhGER2mZ2DmqNdxdXXTriFEkQAJ6IUABpADKJhIFUJYg40lAzwTm/HUG3x24jLe8K2NOX9PZfu39Ncex5+wdTOpaB8PbVNfzqtkdCZCANgQogBRAbfIlv7YUQFmCjCcBPRN4fUUYjl99gPmveeL1JpX03Lvu3a06dAUzd8aaxJfJuq+CkSRgGQQogBRA2UymAMoSZDwJ6JGAKLXiEbAbaSZYcuXMrWR0/eYgCtvZIMrfD/a21npcObsiARLQhgAFkAKoTb7wDqAsLcaTgIEJRMY/RO+locrHFhFfdDKposvZ2TloOjsYiY/TsWl4C3hXK2FgGuyeBEjgRQQogBRA2auDdwBlCTKeBPRI4MdDVzBjZyza1ymDH99tqsee9dPVyPUnsTP6FkZ3qIlPO9XST6fshQRIQGsCFEAKoNZJkyeAAihLkPEkoEcCuYL1mV8tjGxf8AWg8y5tw7E4TNoag6ZVi2PziJZ6XDm7IgES0IYABZACqE2+5NeWAihLkPEkoEcCrb7aixsPn2L90GZoWaOUHnvWT1dxiU/gOz8EttZWynuATg62+umYvZAACWhFgAJIAdQqYfJpTAGUJch4EtATgdtJqWg+Zw+srYCYgM4mKVc5OTloPTdEkdQ1Q5qibe0yelo9uyEBEtCGAAWQAqhNvvAOoCwtxpOAAQn8FXMLH/1yEvXKO+Ov0T4GHEmu6wm/RWFT+HUM83XD5G515TpjNAmQgE4EKIAUQJ0S57kg3gGUJch4EtATgVk7Y7Hy0BUMal4Zs3qbTgHovMvbFnkDozdGon4FZ/z5iemKqp5OC7shAZMkQAGkAMomJgVQliDjSUBPBPosC0VE3EMs7N8AfRtV1FOv+u/m+b2Kxb7AxZ3s9T8IeyQBEngpAQogBVD2EqEAyhJkPAnogUBaZhY8/AORnpWN/ePbokpJJz30argu/Bbtx/mER1g2sBG6eZQ33EDsmQRIIF8CFEAKoOylQQGUJch4EtADgRPXHqDf8jCUdLJH+NSOJlUAOr/lBWw/jTVhV03+cbUeTg27IAGTJEABpADKJiYFUJYg40lADwRWHryMWX+eQce6ZbHynSZ66NGwXQSevo1hP52AWykn7P2srWEHY+8kQAL/Q4ACSAEMAOCfJzPOAaij4fVCAdQQFJuRgCEJfPjzCfx96jYmdKmNj9rWMORQeuk76WkGGs4IRHYOcHhSe5R3KayXftkJCZCAZgQogBRAIYCvAej4XMpkArinWQqBAqghKDYjAUMRELX1RP2/hOQ0/DqsOZq5lTTUUHrtt9fSUETFP8TXrzfAa41N96MVvS6anZGAiRCgAFIAhQD2BuClY05SAHUExzAS0BcBUVRZ7AAidtcQBaAL29voq2uD9jNv11ks23cJfRu6YuEbuv4KMugU2TkJWCwBCiAFUAjgeABJAFIBHAYwCUDcC7LeAYD4J/coCuB6UlISnJ2FC/IgARIwNoEdUTcxakMEPFxdsGNUa2MPr/N4hy7cw6BVR1HOuZDyGNjKykrnvhhIAiSgHQEKIAWwK4AiAMR7f6IWg3gf0BWAO4CUfNIpv3cGQQHU7sJjaxLQJ4HcL2rfaVEF03uJS9c8jtSMLHhOD0R6Zjb2jGuD6qXFryIeJEACxiBAAaQA5s2zYgCuARgLYFU+Scg7gMa4MjkGCWhBoNe3hxB1PQnfvOmFXl7i7zfzOd76/ggOX07EzF71MbhFVfOZOGdKAmZOgAJIAcwvhY8DCH72KPhVKc53AF9FiD8nAQMSEHfR3P13IzM7BwcntEOlEo4GHE3/XX+79wK+DjyPLvXLYcXgxvofgD2SAAnkS4ACSAHMmxjiGYx4/0886v2vBtcNBVADSGxCAoYicPzqfby+4jBKF3XAsckdzO49upNxD9B3WRhcCtvh5BedYGPN9wANlSvslwSeJ0ABpAB+DWDHs8e+FQBMf/ZFcD0AdzW4XCiAGkBiExIwFIEV+y/hq7/PonP9svhusOkXgM7LITMrG14zgvAoLRM7R7WGu6uLoVCxXxIggecIUAApgBsB+AIQhcOE8B0CMAXAJQ2vFAqghqDYjAQMQWDYunAExiZgcrc6GOZb3RBDGLzP99ccx56zdzCpax0Mb2OeazA4JA5AAnomQAGkAMqmFAVQliDjSUBHAqIAdNPZe3DvURp+G9ECTaqW0LGngg1bdegKZu6MhW+t0lj3nnfBToajk4BKCFAAKYCyqU4BlCXIeBLQkUD8/SfwmRcCO5t/CkAXsjOPAtB5l3v2djK6LD6IwnY2iPL3g72ttY5EGEYCJKApAQogBVDTXHlROwqgLEHGk4COBP6IuIExv0bCq1Ix/PFxKx17KfgwcSezyaxgJD5ON6ut7AqeHGdAAroToABSAHXPnn8iKYCyBBlPAjoSmLbtFNYdvob3WlXDtJ7iuy3zPUauP4md0bcwukNNfNqplvkuhDMnATMhQAGkAMqmKgVQliDjSQjE4WIAACAASURBVEBHAj2WHMSpG8n4dkBD9PAUH/Gb77HhWBwmbY1B06rFsXlES/NdCGdOAmZCgAJIAZRNVQqgLEHGk4AOBJ6kZ8IjIBBZ2TkIm9geFYoV1qEX0wmJS3wC3/khsLW2Ut4DdHKwNZ3JcSYkYIEEKIAUQNm0pgDKEmQ8CehA4PClRLz1wxGUdymEw5M66NCD6YW0nrsX1x88xeohTdGudhnTmyBnRAIWRIACSAGUTWcKoCxBxpOADgSWhlzE/N3n0N2jPJYObKRDD6YX8vlv0fg1PB4f+FTDlO7m/U6j6dHljEjg3wQogBRA2WuCAihLkPEkoAOBoWuPI/jMHUztXhdDfdx06MH0QrZF3sDojZGoX8EZf37iY3oT5IxIwIIIUAApgLLpTAGUJch4EtCSgCib0nhWMO4/TsfWj1qiUeXiWvZgms3vpKTCe/YeWFkBJ6d2QnEne9OcKGdFAhZAgAJIAZRNYwqgLEHGk4CWBK7ce4x2X+9TCibHBPjBwdY8C0Dnt2y/RftxPuERlg1shG4e5bUkw+YkQAKaEqAAUgA1zZUXtaMAyhJkPAloSWDLiesYtzkKjasUx5YPLatkSsD201gTdhUDm1XG7D4eWpJhcxIgAU0JUAApgJrmCgVQlhTjSUBPBKb8HoNfjsZZ5McSQbEJ+GBdOKqVckLIZ231RIzdkAAJ5CVAAaQAyl4VvAMoS5DxJKAlgS6LD+Ds7RQsH9gIXS3sMWnS0ww0nBGI7BxYRH1DLU8tm5OA0QhQACmAsslGAZQlyHgS0ILAo7RMeAbsVgTp6OQOKOtcSIto82jaa2koouIf4uvXG+C1xhXNY9KcJQmYGQEKIAVQNmUpgLIEGU8CWhAIvXgPA1cehWuxwgid2F6LSPNpOm/XWSzbdwl9G7pi4Rte5jNxzpQEzIgABZACKJuuFEBZgownAS0ILNlzAQuCzqNngwpY8lZDLSLNp2mu5JZ1dsCRSR1gJerC8CABEtArAQogBVA2oSiAsgQZTwJaEHh39THsO3cX/j3rYUiralpEmk/T1IwseE4PRHpmNoLHtkGNMkXMZ/KcKQmYCQEKIAVQNlUpgLIEGU8CGhLIzs5Bw5lBEB9KbB/ZCp4Vi2kYaX7N3vr+CA5fTsTMXvUxuEVV81sAZ0wCJk6AAkgBlE1RCqAsQcaTgIYELt55hI4L96OQnSgA3Rl2NtYaRppfs2/3XsDXgefRpX45rBjc2PwWwBmTgIkToABSAGVTlAIoS5DxJKAhgU3h8ZjwWzS8q5bAphEtNIwyz2Yn4x6g77IwuBS2w8kvOsHGmu8BmueZ5KxNlQAFkAIom5sUQFmCjCcBDQlM3BKNjcfjMbyNGyZ1rathlHk2y8zKRsMZQUhJy8SOka3hUdHFPBfCWZOAiRKgAFIAZVOTAihLkPEkoCGB3H1yvx/cGH71y2kYZb7Nhq49juAzdzCxax2MaFPdfBfCmZOACRKgAFIAZdOSAihLkPEkoAGB5NQMNJgeiJwcIHxqR5Qq4qBBlHk3WXXoCmbujIVvrdJY9563eS+GsycBEyNAAaQAyqYkBVCWIONJQAMCB87fxds/HkPlEo44MKGdBhHm3+Ts7WR0WXwQhe1sEOXvB3tby/3oxfzPFldgbgQogBRA2ZylAMoSZDwJaEBgUdB5fLPnAnp7VcDiNy2zAHReDDk5OWg6Oxj3HqXj12HN0cytpAak2IQESEATAhRACqAmefKyNhRAWYKMJwENCAxedRQHL9xTXV28URsisCPqJj7pUBNjO9XSgBSbkAAJaEKAAkgB1CRPKICylBhPAhIERAHoBjMCkZKaiZ2jWsPdVT1fxG48FoeJW2PQpEpx/PZhSwmKDCUBEnieAAWQAih7RfAOoCxBxpPAKwicT0iB36IDcLS3QbS/H2wtuAB0XhRxiU/gOz8EttZWynuATg62zBcSIAE9EKAAUgBl04gCKEuQ8STwCgIbjsVh0tYYtHAriQ3DmquOV+u5e3H9wVOsHtIU7WqXUd36uWASMAQBCiAFUDavKICyBBlPAq8gMH5zFDafuI6P21XH+M51VMfr89+i8Wt4PD7wqYYp3eupbv1cMAkYggAFkAIom1cUQFmCjCeBVxDosGAfLt19jFXvNEGHumVVx2tb5A2M3hiJeuWd8ddoH9WtnwsmAUMQoABSAGXzigIoS5DxJPASAg+fpMNrRpDSQuyJW8LJXnW87qakKeVg1MxAdSedCzY4AQogBVA2ySiAsgQZTwIvIRBy9g6GrDkOt1JO2PtZW9Wyyt0Gb9nARujmUV61HLhwEtAXAQogBVA2lyiAsgQZTwIvIbAg8ByW7L2Ifo0qYkH/BqplFbD9NNaEXcXAZpUxu4+Hajlw4SSgLwIUQAqgbC5RAGUJMp4EXkJg4MojCL2YiNl93DGwWRXVsgqKTcAH68JRrZQTQlR8J1S1CcCF650ABZACKJtUFEBZgowngRcQyMrOgWfAbjxOz8Lfo31Qt7y43NR5JKdmwGt6ILJzgLCJ7VGhWGF1guCqSUBPBCiAFEDZVKIAyhJkPAm8gEDszWR0++9BFHGwVYog21hbqZpV76WhiIx/iPmveeL1JpVUzYKLJwFZAhRACuDzOTQRwBwA3wAYo2FyUQA1BMVmJKAtgZ+PXMPUP06hdY1S+HloM23DLa79vF1nsWzfJfRt6IqFb3hZ3Pq4IBIwJgEKIAUwN9+aAtgEIBlACAXQmJchxyKB/AmM3RSJrSdv4JP2NTDWr7bqMYVevIeBK4+irLMDjkzqACsrdd8RVX1CEIAUAQogBVAkUBFRYgzARwCmAoikAEpdVwwmAb0QaDs/BFcTn3ALtGc0UzOy4Dk9EOmZ2Qge2wY1yohfXTxIgAR0IUABpACKvFkL4D6ATwHse4UAOgAQ/+QeRQFcT0pKgrOzel9Q1+XiYwwJvIxA4qM0NJ71T/HjqGl+cHG0IzAAA344grBLiZjRqz7eblGVTEiABHQkQAGkAL4JYAoA8Qg4VQMBDADgnzffKIA6XoEMI4EXEAiOTcDQdeHKXS5xt4vHPwSWhlzE/N3n0Ll+WXw3uAmxkAAJ6EiAAqhuARSf0YUD6AQg+lkO8Q6gjhcTw0hAnwRyP3jo36Qi5r2m3gLQeZmejHuAvsvC4FLYTtkaT+1fRusz59iXughQANUtgL0B/A4g67m0twGQAyD72aPe53+W39XBr4DV9TuDqzUSgTe+O4yjV+7jq74eeNO7spFGNf1hMrOy0XBGEFLSMrFjZGt4VHQx/UlzhiRgggQogOoWQPH+Xt6tBVYDOAtgLoBTGuQsBVADSGxCAtoQEJLjERCIpxlZCPrUFzXLikuVRy6BoWuPI/jMHUzsWgcj2lQnGBIgAR0IUADVLYD5pcyrHgHnjaEA6nDhMYQEXkbg1I0k9FhyCM6FbBE5zQ/WKi8AnZfVj4euYMbOWPjULIWf3md9RF5NJKALAQogBTBv3lAAdbmSGEMCeiSw7vBVTNt2Gr61SmPde9567Nkyujp3OwWdFx9AITtrZYcUB1vx5goPEiABbQhQACmA2uRLfm15B1CWIONJIA+B0RsjsC3yJsZ0rIkxHWuRTx4COTk5aDo7GPcepePXYc3RzK0kGZEACWhJgAJIAdQyZf6nOQVQliDjSSAPAZ95exF//yl+et8bPjVLk08+BEZtiMCOqJv4pENNjO1ESWaSkIC2BCiAFEBtcyZvewqgLEHGk8BzBO6mpCl3t8QuZ9H+fihaiAWg80uQjcfiMHFrDJpUKY7fPmzJHCIBEtCSAAWQAqhlyvAOoCwwxpPAywjsPn0bw386gdpli2L3p76E9QIC8fefwGdeCGytrZT3AJ0cbMmKBEhACwIUQAqgFumSb1PeAZQlyHgSeI7AnL/O4LsDl/GWd2XM6etBNi8hkPuofPW7TdGuThmyIgES0IIABZACqEW6UABlYTGeBF5F4PUVYTh+9QHmv+aJ15uIzXp4vIjAxC3R2Hg8Hh/4VMOU7vUIigRIQAsCFEAKoBbpQgGUhcV4EngZgfRMUQB6N9Iys7FnXBtUL12EwF5CYFvkDYzeGIl65Z3x12gfsiIBEtCCAAWQAqhFulAAZWExngReRiAq/iF6LQ1FMUc7RHzRCVbiSxAeLySQ+8GMaCD2BS7hZE9aJEACGhKgAFIANUyVFzbjO4CyBBlPAs8I5O5w0b5OGfz4blNy0YBA50UHcC4hBUsHNEJ3z/IaRLAJCZCAIEABpADKXgkUQFmCjCeBZwRGrj+JndG38JlfLYxsX5NcNCAwfcdprA69igHNKuPLPvxoRgNkbEICCgEKIAVQ9lKgAMoSZDwJPCPQ6qu9uPHwKdYPbYaWNUqRiwYEgmMTMHRdOKqVckLIZ201iGATEiABCuA/OcCXbOSuBQqgHD9Gk4BC4HZSKprP2QNrKyAmoDPr2mmYF8mpGfCaHojsHCBsYntUKFZYw0g2IwF1E+AdQAqg7BVAAZQlyHgSAPBXzC189MtJftGqQzb0XhqKyPiHLJ2jAzuGqJcABZACKJv9FEBZgownAQCzdsZi5aErGNS8Mmb15rts2iTF/N1nsTTkEvo0dMWiN7y0CWVbElAtAQogBVA2+SmAsgQZTwIA+i4Lxcm4h1jYvwH6NqpIJloQCLt4DwNWHkWZog44OrkDy+dowY5N1UuAAkgBlM1+CqAsQcarnkBaZhY8/AORnpWNfZ+1RdVSTqpnog2A1IwseE4PhCikHTy2DWqUYQFtbfixrToJUAApgLKZTwGUJch41RM4ce0B+i0PQ0kne4RP7cg7WDpkxIAfjiDsUiJm9KqPt1tU1aEHhpCAughQACmAshlPAZQlyHjVE1h58DJm/XkGHeuWxcp3mqiehy4AloZcxPzd59C5fll8N5gMdWHIGHURoABSAGUzngIoS5Dxqifw0S8n8FfMbUzoUhsfta2heh66AIiIe4A+y8LgXMgWEdP8YCPq6fAgARJ4IQEKIAVQ9vKgAMoSZLyqCeTk5Cj1/xKS07BxWHM0dyupah66Lj4zKxsNZwQhJS0T20e2gmfFYrp2xTgSUAUBCiAFUDbRKYCyBBmvagJi5w+xA4i4Y3UqoDMK29uomofM4oeuDUfwmQRM7FoHI9pUl+mKsSRg8QQogBRA2SSnAMoSZLyqCeyIuolRGyLg4eqCHaNaq5qF7OJ/PHQFM3bGwqdmKfz0fjPZ7hhPAhZNgAJIAZRNcAqgLEHGq5rA9B2nsTr0Kt5pUQXTe7mrmoXs4s/dTkHnxQdQyM4aUf5+cLDl3VRZpoy3XAIUQAqgbHZTAGUJMl7VBHp9ewhR15PwzZte6OXlqmoWsosX71M2nb0H9x7xfUpZloy3fAIUQAqgbJZTAGUJMl61BEQBY3f/3cjMzsHBCe1QqYSjalnoa+GfbIjA9qib+KR9DYz1q62vbtkPCVgcAQogBVA2qSmAsgQZr1oCx6/ex+srDqN0UQcc4xZmesmDX4/H4fMtMWhSpTh++7ClXvpkJyRgiQQogBRA2bymAMoSZLxqCXy3/xLm/H2WxYv1mAHx95/AZ14IbK2tlPcAnRxs9dg7uyIByyFAAaQAymYzBVCWIONVS2DYunAExiZgUtc6GM6yJXrLA595exF//ylWv9sU7eqU0Vu/7IgELIkABZACKJvPFEBZgoxXJYHnP1j4bUQLNKlaQpUcDLHoiVuisfF4PIa2roapPeoZYgj2SQJmT4ACSAGUTWIKoCxBxquSQO6jSjsbK8QEdEYhO5Ys0VciKB+BbIhA3fLO+Hu0j766ZT8kYFEEKIAUQNmEpgDKEmS8Kglsi7yB0Rsj0aBSMWz7uJUqGRhq0XdT0tB0drDS/ckvOqGEk72hhmK/JGC2BCiAFEDZ5KUAyhJkvCoJTNt2CusOX8N7raphWk8+ptR3EnRedADnElKwdEAjdPcsr+/u2R8JmD0BCiAFUDaJKYCyBBmvSgI9lhzEqRvJ+HZAQ/TwrKBKBoZcdO4OKwOaVcaXfTwMORT7JgGzJEABpADKJi4FUJYg41VH4El6JjwCApGVnYOwie1RoVhh1TEw9IKDYxMwdF04qpZ0xL7x7Qw9HPsnAbMjQAGkAMomLQVQliDjVUfgyOVEvPn9EZRzLoQjkzuobv3GWHBKaga8ZgQpkh06sT1cKdnGwM4xzIgABZACKJuuFEBZgoxXHYGlIRcxf/c5dPcoj6UDG6lu/cZacJ9loYiIe4h5r3mif5NKxhqW45CAWRCgAFIAZROVAihLkPGqIzB07XEEn7mDqd3rYqiPm+rWb6wFz999FktDLqFPQ1csesPLWMNyHBIwCwIUQAqgbKJSAGUJMl5VBEQB6MazgnH/cTq2ftQSjSoXV9X6jbnYsIv3MGDlUZQp6oCj3GvZmOg5lhkQoABSAD8EIP6p+ixfTwOYAeBvDfOXAqghKDYjAUHg6r3HaPv1PtjbWCNmuh8cbFkA2lCZkZqRhQbTA5GWmY3gsb6oUaaooYZivyRgdgQogBTAngCyAFzAPyzeATAeQEMAQgZfdVAAX0WIPyeB5whsOXEd4zZHoXGV4tjyYUuyMTCBgSuPIPRiIqb/pz7eaZn7d66BB2X3JGAGBCiAFMD80vT+MwlcpUEOUwA1gMQmJJBLYMrvMfjlaBw+8KmGKd1ZANrQmZH7wY1fvbL4/u0mhh6O/ZOA2RCgAFIAn09W8SzqdQBrn90BjM0nkx0AiH9yD/FM5XpSUhKcnYUL8jA2gesPnuDnI3Ho6l5O2VaMh2kT6PrNQZy5lYzlAxuhqwd3qDD02YqIe4A+y8LgXMgWEdP8YGMtHnTwIAESoABSAMVVIMrkHwZQCMAjAAMA/PWCyyMAgH/en1EAjf/LRNQ3Wxt2FV8HnsOT9Cylpty+8W1RyI7vlBn/bGg24qO0THgG7EZ2DpSPEso6i0uOhyEJZGZlo+GMIKSkZWL7yFbwrMg/kgzJm32bDwEKIAVQZKvYKb0yABcArwEYCqANAN4BNNFrWdxBmrg1BlHxD5UZipsaQipYVsRET9izaYVevIeBK48qRYlFcWIexiEwdG04gs8k4PMudfBh2+rGGZSjkICJE6AAUgDzS9FgAJcADNcgf/kOoAaQ9NVEfNW4ZO8FfLf/MjKzc1DUwRYTu9WBjZWVIoQlnOxxYEI7FHGw1deQ7EePBJbsuYAFQefRs0EFLHlLfGfFwxgEVodewfQdsfCpWQo/vd/MGENyDBIweQIUQApgfkm6F0AcgHc1yGAKoAaQ9NFEbB82aWsMrtx7rHTXpX45TO9VX3mMKB5z+S06gMv3HmNsp1r4pENNfQzJPvRMYMjqYwg5dxf+PethSKtqeu6d3b2IwPmEFOX6KGRnjSh/lt5hppCAIEABpADOeVbzTwif+KBDvP/3OYDOAII0uEwogBpAkmmS9CQDc/4+g43H45VuRFHbGb3c0cW93L+63RF1E6M2RCh3BQ9+3g7FHMWTfR6mQiA7OwcNZwYh6WkGtn3cih/sGPHEiOLbTWfvwb1Hadg4rDmau5U04ugcigRMkwAFkAIoSr2I3ejF54hJAKIBzNVQ/kRWUwANdG2Lf2n9feo2/Lefxt2UNGWUAc0qK+8xuRS2+59RhWB0X3JI+cJ0RJvqmNi1joFmxm51IXDxziN0XLhfuQsVE9AZdjbWunTDGB0JfLIhAtujbuKT9jUw1q+2jr0wjAQshwAFkAIom80UQFmC+cTfTkrFF9tOISg2QfmpW2knfNXXE97VSrx0tD1nEvD+2nBFMg6Mb4cy/MrUAGdHty43hcdjwm/R8K5aAptGtNCtE0bpTODX43H4fEsMC3DrTJCBlkaAAkgBlM1pCqAswefixV28X47FYd7fZ5WyFbbWVspXix+3q6FReRdx17Df8jCcjHuIt1tUUR4V8zANApO2RmPDsXgMb+OGSV3rmsakVDSL+PtP4DMvRLmmIv39+KGUis49l5o/AQogBVD22qAAyhJ8Fn/xTgombolB+LUHyv/jVakYvurngTrltCuwHXbpHgb8cBR2NlbYO64tKpVw1NMM2Y0MAb9F+3E+4RG+G9wYnev/+/1NmX4ZqzkB33khiLv/BD++2wTt65TVPJAtScACCVAAKYCyaU0BlCSYlpmF5fsuYVnIJaRnZcPJ3gbjO9fG4BZVdd61YNDKozh08R5ea1wRX7/eQHKGDJclkJyagQbTA5GTAxyf0hGliz6/mY5s74zXlMDELdHKx1RDW1fD1B7chk9TbmxnmQQogBRA2cymAEoQPHHtvnLX78IdsQEL0L5OGczs7a4UCpY5IuMfovfSUKVAdOCnbVCjTBGZ7hgrSeDA+bt4+8djqFzCUanTyKNgCCgfgWyIQN3yzvh7tE/BTIKjkoCJEKAAUgBlU5ECqAPBlNQMzN99Dj8duabcFSpVxB7+Peujh2d5WFnpZ6/SD9aFKx+RdPcoj6UDG+kwS4boi8Di4PNYHHwBvb0qYPGbLACtL67a9iPKwDSZJercAyemdkTJIrwTqy1DtrccAhRACqBsNlMAtSQYHJuAqX+cwu3kVCXy9cYVMaV7Xb3X7Tt7OxldvzmoCObOUa3h7ip2+uNREAQGrzqKgxfuYUav+ni7RdWCmALHfEagy+IDOHs7Bd8OaIgenhXIhQRUS4ACSAGUTX4KoIYE76SkKttR/Rl9S4kQjwO/7OOB1jVLadiD9s1Gb4zAtsibaFe7NFYP8da+A0ZIExBfdjeYEYiU1EyKuDRN+Q5m7IjFj6FX8JZ3Zczp6yHfIXsgATMlQAGkAMqmLgXwFQRFaZbN4dcx689YJKdmKh92DPWphjEdaqGwvY0s/5fGX733GB0W7kdWdg5+G9ECTaq+vI6gQSej0s5ztyFztLdBtL8fbFkAukAzQdyBH7ouHFVLOmLfeL6PWaAng4MXKAEKIAVQNgEpgC8hKPbtnbw1BocvJyqt3F2dlYLOxnwcK/YP3nAsTiki/euw5np7x1A2cdQSv/FYHCZujUFztxLYOIwFoAv6vIv3b71mBCl/FIVObC/9wVVBr4fjk4CuBCiAFEBdcyc3jgKYD8GMrGz8cPAyvgm+gLTMbGVnjnGdamNIq6pGvwN0K+kp2szfh/TMbKx7zxu+tUrLnnPGa0Fg/OYobD5xHR+3q47xnbk9nxboDNa0z7JQRMQ9xLzXPNG/SSWDjcOOScCUCVAAKYCy+UkBzEMwKv6hcsdH7MkrDp+apTC7twcqlyy4gswzd8Zi1aEr8Kzogm0ft+JdQNms1yK+w4J9uHT3MVa90wQd6rL4sBboDNb0693n8G3IRX6VbTDC7NgcCFAAKYCyeUoBfEbwSXomFgaeV14wz84Bijna4Yvu9dC3kWuBC5cofyF2QXiSnoUVgxqjizt3opBNfE3iHz5JVx43iuPkF51QwslekzC2MTCB3N1yREHuY5M7FPj1aeDlsnsSyJcABZACKHtpUAAB7D9/F1N+j8H1B08Vnr28KuCLHvVQyoTqjC0IPIcley+iZpki2DXGV+ddRmQTRk3xIefuYMjq46hWygkhn7VV09JNeq2pGVnKzizi9Yzgsb6oUaaoSc+XkyMBQxCgAFIAZfNK1QJ4/3E6xOPV3yNuKBzFDh6z+rijXe0yslz1Hp/0NEO5Cyj+c9EbDdCnYUW9j8EO/00gV7r7NaqIBf25JZ8p5cfAlUcQejER0/9TH++0ZG1GUzo3nItxCFAAKYCymaZKARSlXf6IvAFRU+zBkwyIzTuGtKyGcX614ORgK8vUYPHL9l3EvF3nlBqEwWPbwN7W2mBjsWMgVzJm93HHwGZViMSECCwNuajsxuNXryy+f7uJCc2MUyEB4xCgAFIAZTNNdQIYf/8JpvxxCmJ/V3HUKVcUX/XzhFelYrIsDR4v3lP0nbcP4p3AWb3dMag5pcRQ0EWZEc+A3XicnqXsOyv2n+VhOgRy98t2LmSLiGl+fCXCdE4NZ2IkAhRACqBsqqlGADOzsrEm7CoWBJ7H04ws5e7Z6A41MczXDXZmVNx3TegVBOyIRVlnB+wf3w6F7AxbjFo2wcw1XnwFLrbiK+Jgiyh/CoapnUdxPTecGaTs0CK+jG9gBn/AmRpDzse8CVAAKYCyGawKAYy9mYyJW6MRfT1J4dWsWgllGym30kVk+Rk9Pi0zC+2/3o8bD59iSre6+MDXzehzUMOAPx+5puz53LpGKfw8tJkalmx2axy6NhzBZxLweZc6+LBtdbObPydMAjIEKIAUQJn8EbEWLYDia8Fv9lzA9wcuKzsHFC1ki8nd6uKNJpVgbW0ly67A4jeFx2PCb9Eo7miHAxPaoWghuwKbi6UOPHZTJLaevIFP2tfAWL/alrpMs17X6tAryv7colbnT+9T0s36ZHLyWhOgAFIAtU6aPAEWK4CiVpjYxu1q4hNlyV3dyylfDJZxLiTLrMDjxeMvv8UHcPnuY3zasRZGd6xZ4HOytAm0+3ofxFaAq4c0Ncmvwi2Nty7ryd2nWezUIx7TO9jydQhdODLGPAlQACmAsplrcQKY9CQDs/+Kxabw6wob8a7czF7u8KtvWcWTd0bfxMj1Eco7agcntENxFimWvRb+f3ziozQ0nhWs/O+oaX5wceQdVr3B1WNH4mt+7y/34G5KGjZ80BwtqpfUY+/sigRMmwAFkAIom6EWI4DiXwZ/xtxCwPZY5StZcQxqXhkTutSBswU+Is3OzkH3JYeULeuGt3HDpK51ZXOB8c8IBMcmYOi6cNQoU0Qpt8PDdAmM3hiBbZE3Map9DYzjo3rTPVGcmd4JUAApgLJJZRECePPhU0zbdgrBZ+4oPKqXdlJKuzStWkKWj0nH7z2bgPfWhEM8AhNfBJe1gMfbpgB83q6zWLbvEvo3qYh5r7EAtCmckxfN4dfjcfh8SwwaVymOLR+2NOWpcm4koFcCFEAKoGxCmbUAirtgPx+9ueeRfgAAIABJREFUhrl/n1XqtdnZWOHDtjXwcbvqqngfSNz1fG3FYZy49gCDm1fBzN7usvnAeABvfn8YRy7fx1d9PfCmd2UyMWECoq6nz7wQ2FpbIdLfT3klggcJqIEABZACKJvnZiuAFxJSMHFrjCI/4mhUuZhy169WWXXtC3r4UiLe+uGIIr97x7VFpRKOsjmh6njxgY1HQKBSKzLwU1/V5ZM5nnyxRWLc/Sf48d0maF+nrDkugXMmAa0JUAApgFonTZ4AsxNAUQdvWcgliG3RMrJy4GRvg8+71sGgZlXMurSLzIkcvOooDl64B+5ZK0Pxn9hTN5LQY8khpWSQ+ADEnMsFydMwjx4mbY3GhmPxeL91NXzRo555TJqzJAFJAhRACqBkCplXHcDwq/eVu34X7zxS1t2xbhnM6OWOCsUKy3Iw6/jcbbFEaUNx16pGGXXdBdXnyVt3+CqmbTsN31qlse49b312zb4MRGBH1E2M2hChbOu4a4yvgUZhtyRgWgQogBRA2Yw0izuAyakZEC/m/3wkTllvqSL2mP4fd3TzKAcrK/Mt6Cx78p6PH7YuHIGxCQqTZQMb67NrVfU1ZmME/oi8iTEda2JMx1qqWru5LlZ89d/kWdmeE1M7omQRB3NdCudNAhoToABSADVOlhc0NHkBDDx9G19sO4WE5H9Ku4gvM8VuHsUc7WXXblHx526noMs3B5CTA+wc1Rruri4WtT5jLcZn3l7E33+q3P0TdwF5mAeBLosP4OztFHw7oCF6eFYwj0lzliQgQYACSAGUSB8l1GQF8E5yKgJ2nMZfMbeViVYp6Yg5fTzQskYp2TVbbHzu3au2tUtjzRA+vtT2RIuCwk1nB0PcVBY7S1hi/UhtmZhL+xk7YvFj6BW85V1Z2eebBwlYOgEKIAVQNsdNTgBFaZNfj8dj9l9nkJKaCRtrKwzzdcPoDjVRyI5bPb3shF+99xgdF+5HZnYONo9oYfF1EGWTP2/87tO3MfynE6hdtih2f8p3yfTN15D97TmTgPfXhit/KIqamDxIwNIJUAApgLI5blICePnuI0zaGoOjV+4r6/Ks6KL8NV+/Ah9nanqiBb8Nx+LgXbUEfh3enO9IagoOwJy/z+C7/ZfxlnclzOnrqUUkmxY0gZTUDHjNCEJWdg4Ofd4OFYuzHFJBnxOOb1gCFEAKoGyGmYQAZmRl4/sDl/HNngtIz8xGYTsbjPOrhXdbVoWtjbXsGlUVfyvpKdrM36dwXPueN9rwPTaNz//rK8Jw/OoDzHvNE/2bVNI4jg1Ng0CfZaGIiHvI82cap4OzMDABCiAFUDbFClwARQmTiVuilRe4xeFTsxS+7OPBgsYSZ3bWzlisPHQFHq4u2D6yFe8CasBSCLNHwG6kZWZjz7g2qF66iAZRbGJKBL7efQ7fhlxEb68KWPxmQ1OaGudCAnonQAGkAMomVYEJ4OO0THwdeA5rwq4qX64Wd7TDtJ710NvLlcIieVYTH6VB7I4gtsdbMagRuriXl+zR8sOj4h+i19JQFHO0Q8QXnZiDZnjKwy7dw4AfjqJ0UQccm9yB59AMzyGnrDkBCiAFUPNsyb9lgQhgyLk7mPr7Kdx4+FSZVZ+GrpjavS7rd8mezefiFwaew3/3XkSNMkWwe4yv8jENjxcTWB16BdN3xKJd7dJYzS+ozTJVUjOy0GB6oHIXN+hTX9RU2baQZnnSOGmdCVAAKYA6J8+zQKMKoLgzNWNnLLZF3lSGdy1WGF/29eB7arJnMZ94UTzbZ24Ikp5mYGH/BujbqKIBRrGcLkeuP4md0bfwmV8tjGxf03IWprKVDFp5FIcu3kNAz3p4t1U1la2ey1UTAQogBXASgL4A6gAQt9PCAHwO4JyGF4JRBFCUdtl68gZm/RmLB08yIG5GDWlVDWM71YKTg62GU2UzbQks33cJc3edRaUShbFnbFvY2/KDmhcxbPXVXuWO9PqhzVhrUttEM6H2S0MuYv7uc/CrVxbfv93EhGbGqZCAfglQACmAuwBsBHAcgDCpLwG4AxA7oj/WIN0MLoBxiU8w5Y8YHLxwT5mO2K9zbj9PNKhUTIPpsYkMgSfpmcoXwaLA8cze7hjcvIpMdxYbm5CcimZf7lH+MIkJ6Mw/Ssz4TOfui+1cyBYR0/z46oMZn0tO/eUEKIAUwLwZIvauugOgDYADGlxABhPAzKxspTL/wqDzSM3IVu4+if1VP/Bxgx1Lu2hwavTTZG3YVfhvP40yRR1wYEI7FtPOB+vfMbfw4S8nUbe8M/4e7aMf8OylQAiIOoBeMwKVIvLbPm7FPzQL5Cxw0OcJiCdghtizngJIAcx7pdUAcAGA2AvpVD6Xodgl/fmd0osCuJ6UlARnZ+GC+jlO3UjCxK3ROHUjWemwuVsJpbButVJO+hmAvWhMIC0zC+2/3q883pzcrQ6G+VbXOFYtDXPL5gxqXhmzenMbMXM/7x+sC0dQbAImdKmNj9qKX4k8SKBgCBy/eh/+204re1S76bm0FAWQAvh8VosXvLYDEM9WW78g3QMA+Of9mb4FcNymKGw5eR3iMcyU7nWVorqG+AuoYC5p8xt1U3g8JvwWrZTaEXcBixayM79FGHDGfZeF4mTcQ34sY0DGxux6TegVBOyIResapfDz0GbGHJpjkYBCQHyEN/fvs/jlaJzyv7t5lMOygY31SocCSAF8PqGWA+j6TP6uvyDTjHIH8P7jdCX5x3WuhTJFC+k16dmZ9gTE43i/xQdw+e5j5TH8mI61tO/EQiPEHVIP/0CkZ2Vj32dtUZV3qc3+TJ9PSIHfogNwsLVGdIAfHGy5h7jZn1QzWsCuU7fhv/0UEpLTlFm/0aQSJnerCxdH/f7hTQGkAOZeFt8C6AVA7GB/RYtrxWDvAGoxBzY1AoE/o2/h4/UnUcTBFgcntENxJ3sjjGr6Q5y49gD9loehhJM9TkztyDvVpn/KXjlD8c6V95d7lI+fNnzQHC2ql3xlDBuQgCwB8TGZeNy76/RtpSvxypPY1cpQ+UcBpACK6r5LRC1lAG2fvf+nTR5TALWhZcZts7Nz0GPJIcTeSsZwXzdM6lbXjFejv6mvPHgZs/48g451y2LlOywboj+yBdvT6I0RSr3RUe1rYJxf7YKdDEe3aALid+vG4/GY8/cZ5eMjW2srDPN1wycdahr0ozsKIAVwGYABz+7+PV/7L+lZXcBXXXgUwFcRsqCfh5y9gyFrjiuPxsS7gGWd+Xj+o19O4K+Y2/xgwILyXCxl0/F4TNgSjUaVi2HrR60sbHVcjqkQuHT3ESZtjcGxK/eVKTWo6KJ88Fivgv4+qnzRWimAFMCcFyTHEABrNLhIKIAaQLKUJuLR2GsrDkM89hQ1AUVtQDUfgkfzOXuUd3U2DmuO5m58VGgp+XD9wRO0nhui1AGMnNaJHz5Zyok1kXWkZ2bj+wOXlO02xX8vbGeDzzrXxrstqxqt9iQFkAIoezlQAGUJmln8kcuJePP7I8pjir3j2qJySUczW4H+pitK44gdQIQkxAT4wdGeu9Loj27B9+Q7LwRx95/gx3eboH2dsgU/Ic7AIghExD3AxC0xOJeQoqzHt1ZpzO7tjkoljPu7lAJIAZS9oCiAsgTNMH7wqqPKzix9G7liYX8vM1yBfqa8I+omRm2IgIerC3aMelHlJP2MxV6MT2DS1mhsOBaP91tXwxc9xOZIPEhAdwKP0zKVbQbXHr6KnBwoH45N61EPvbwqFMjHYxRACqDu2fxPJAVQlqAZxkfFP0SvpaHK1me7x/iiZllRD1x9x/Qdp7E69CreaVEF03up+3G4JZ79XMEX20/uGiMKJPAgAd0IiPenp/5xSimoL46+DV0xtUc9RQIL6qAAUgBlc48CKEvQTOOH/xSO3acT0NW9HJYP0m+BUnNBIiRYyPA3b3qhl5eruUyb89SQQOKjNDSeFay0Dp/aEaWKPL8JkoadsJmqCdx7lIYZO2KxPeqmwqFi8cJKaRfx2LegDwogBVA2BymAsgTNNF4Uy+28+IDyKGPHyNbwqOhipivRbdqpGVlw99+NzOwcpS6isd/f0W3WjNKWQJfFB3D2dgqWvNUQPRtU0Dac7VVKQHwgtuXkDcz6MxYPn2QoT0vEqwSfdqplMu8KUwApgLKXJwVQlqAZx3/6ayR+j7iBNrVKY+173ma8Eu2nLvbofH3FYZQu6oBjkzsUyDs82s+aEdoSEHdvfgy9gre8K2NOX+7zrC0/Nba/lvgYU34/hUMX7ynLr1veGXP7ecCzothl1XQOCiAFUDYbKYCyBM04Xvyi67Bgv3IXbNPwFvCuVsKMV6Pd1L/bfwlz/j6LzvXL4rvBLACtHT3zab3nTALeXxuOKiUdsX98O/OZOGdqdAJiy0zxx8LCoPNIzchW6qWKbTOH+lSDnY210efzqgEpgBTAV+XIq35OAXwVIQv/+eTfY7D+aByaVi2uSKCVldhcxvKP3HcgJ3Wtg+Ftqlv+glW6wpTUDHjNCEJWdg4Ofd4OFYsbt1SHSrGb3bJP3UjCxK3ROHUjWZl7C7eSyh1jU94bnAJIAZS90CiAsgTNPP52Uip854coxUzXDGmKtrXLmPmKXj198X5P09l7IF7w3jyiBZpWVc+dz1fTsbwWfZeF4mTcQ8zr54n+TStZ3gK5Ip0JPE3PwuLg81h56IryR4JLYTtM6V4XrzeuaPJ/DFMAKYA6J/6zQAqgLEELiJ+1M1b5Beju6qx8EGLpdwHj7z+Bz7wQ2NmIAtCdDbpfpwWkh9kvYUHgOSzZe1Gp1/bNmw3Nfj1cgH4IHLpwD+IJiCgWLo4enuXh37O+8l6wORwUQAqgbJ5SAGUJWkC8KJchdk14nJ6F5QMboatHeQtY1YuXsC3yBkZvjESDSsWw7WPuE2vRJxtA2KV7GPDDUX7wY+knWsP1PXicjll/nsGWk9eViPIuhTCrtzs61DWv3WIogBRADVP+hc0ogLIELSRevPj83z0XUKNMEaU4tNgezVIP/22nsPbwNQxpVVX5i5+HZRMQJX8aTA9EWmY2gj5Vb+Fzyz7Lr16dePVjR/QtTN9+GomP0yFed367eRWM71IHRRzMbxtICiAF8NVZ//IWFEBZghYSn5yaAZ+5IUh6moEFrzdAv8YVLWRl/7uMHksOKi97fzugIXp4sjacxZ7o5xY2aOVRpaxHQM96eLdVNTUsmWt8joDYwWPq7zEIOXdX+X9rlimCr/p5onGV4mbLiQJIAZRNXgqgLEELil+x/xK++vusUu1+77i2sLc1vdIHsrifpGfCIyBQeeE7bGJ7VChWWLZLxpsBgWX7LmLernPoVK8sfnibZX/M4JTpZYriOl93+Kqyh++T9CzY21jj43Y18GHb6mb/+40CSAGUvUgogLIELShefBEnvgi+m5KGmb3dMbh5FQta3T9LOXI5EW9+fwTlnAvhyOQOFrc+Lih/Arn7XxctZIuILzrB1gTruvHc6ZfA2dvJmLglBpHxD5WORakrUdqlRhnL2PucAkgBlL1iKICyBC0sXvy1PG3baZQp6qAUzi1sb2NRK8y9E9TNoxyWDVTnHsgWdUI1XIy4E+Q1IxApqZnKhz/iAyAelklAvPP57d6LEE80RJH7og62+LxrHQzwrgxrC3q3mQJIAZS9gimAsgQtLF7UA2y/YB+uP3gKSyySPHTtcQSfuYOp3etiqI+bhZ09LudlBD5YF46g2ARM6FIbH7WtQVgWSODo5URM2hqDy/ceK6vzq1cWM3q5o5xLIYtbLQWQAiib1BRAWYIWGL85PB7jf4tGMUc7HJzQDkUL2VnEKsVXgI1nBeP+43Rs/aglGlU23xfALeKEGHkRa0KvIGBHLFrXKIWfhzYz8ugczpAExMdr4v3lDcfilGFELb+Zveqji7vllrSiAFIAZa8pCqAsQQuMF3tidl58AJfuPsboDjXxaadaFrHKq/ceo+3X+5QXwWOm+8HB1rIeb1vESTLgIi4kpKDTogPKHq9R/n4sAG5A1sbsetepW8prK3dS0pRh3/KuhIld6yq7eljyQQGkAMrmNwVQlqCFxv8ZfQsfrz+p1Mc6MKEdSjjZm/1Kt568jrGbotCocjFs/YgFoM3+hGq5AHEHuNmXexRRWP9BM7SsXkrLHtjclAgkJKdi2rZT2H06QZmWWyknfNnXA83dSprSNA02FwogBVA2uSiAsgQtND47Owc9vz2E0zeTMczXDZO71TX7lU75PQa/HI3DBz7VMKV7PbNfDxegPYExGyPwR+RNjGpfA+P8amvfASMKnID43bT+WBzm/n0WKWmZsLW2wog21TGyfQ1V3dWlAFIAZS9GCqAsQQuODzl3B0NWH1cemYm7gGWdzftF6q7fHMSZW8mq2O7OgtNSammbjsdjwpZo3gWWolhwwRfvPMKkrdE4fvWBMgnxNffcfh6oU078q0xdBwWQAiib8RRAWYIWHC8emb2+4jDCrz3AoOaVMau3h9mu9lFaJjwDdiM7Bzg6uYPZy6zZnogCnvj1B0/Qem6IstVh5LROFvOBUwFjNfjwojqBKOsiyrukZ2XD0d4G4zvXxtstqlr0tpUvA0sBpADKXngUQFmCFh4vyiq88f0R5TGL2B2kcklHs1xx6MV7GLjyKFyLFUboxPZmuQZOWj8E2swPwbXEJ1j1ThN0qFtWP52yF4MROBn3ABO3RON8wiNljLa1S2NWb3dULG6ev4v0BYoCSAGUzSUKoCxBFcQPXnUUBy/cQ99GrljY38ssV7xkzwUsCDqPng0qYMlbDc1yDZy0fgiIOnGiXMh7raphWk++C6ofqvrvRdy1/3r3Oaw9fBU5OUBJJ3vlfP2nQQVYWVnpf0Az65ECSAGUTVkKoCxBFcRHX3+I/3wbCvE7d/cYX9Qqa35bKQ1ZfUzZCN6/Zz0MaVVNBWeNS3wRgZ3RNzFyfQTqlCuKXWN8CcoECew9m4Cpv5/CzaRUZXb9GlVUircXt4BqBPrCTQGkAMrmEgVQlqBK4kf8dAK7Tt9Gl/rlsGKweW2hJr4abDQrCA+fZHAbMJXk68uWmfgoTSkILo7wqR1RqogDqZgIAbEP+YydsdgRdVOZUaUShfFlHw/41CxtIjM0nWlQACmAstlIAZQlqJL48wkpSnFo8Shm+8hW8KxoPnupii8HOy7cr3zNHBPQGfa21io5a1zmiwh0WXwAZ2+nKK8DiNcCeBQsAfHB2eYT1zH7zzMQu3qILXs/8HHDmI61LG4/cn2RpgBSAGVziQIoS1BF8WN/jcTWiBvwrVUa697zNpuVbwqPx4TfouFdtQQ2jWhhNvPmRA1HYObOWKw6dEXZNWJOX0/DDcSeX0ngWuJjTP49BqEXE5W29Ss4Y24/T7i7urwyVs0NKIAUQNn8pwDKElRRfFziE7RfsA+Z2Tn4dVhzNDOTivuibtiGY/EY3sYNk7qaf0FrFaWcwZYq3jF7b004KpdwVGpc8jA+AbHl5MpDV7Ao6DzSMrOVO/RjO9XC+62rwdaGd+lfdUYogBTAV+XIq35OAXwVIf78XwRyd9NoWrU4Ng1vYRZf43VedADnElLw3eDG6Fy/HM8oCUB8YdpgeiCysnNwcEI7VCqh7pIixk6JUzeS8PmWaGWnIXG0qlFSedevSkknY0/FbMejAFIAZZOXAihLUGXxt5NSIeqoib/YVw9pina1y5g0geTUDOVf9OLdxeNTOqJ0Ub7wb9InzIiT67ssFCfjHmJeP0/0b1rJiCOrd6in6VlYFHweKw9eVoqyuxS2U77ufa1xRbP4Y9KUzhwFkAIom48UQFmCKoyf/Wcsfjh4RXlXZ8fI1rAWb2yb6HHg/F28/eMxPuoz0fNTkNNaEHgOS/Ze/H/t3QmUFNW9x/Efw7CvCoiALAoIiiwaEJBdBCRRJ/IUjZ6IvhglEgwiuDwxLD6BEIMYlxfUyFM08WhCMCYgyDpIICDovAEUEEQYVECHTfZZ3vm3PYLAMNVzq6erpr91jud44N5btz/17+E3tdxSWrv6eupm1oaM97FYsnFX5F6/bdmHIruy9fxsXT+ewi6ePAGQAFi8yjneiwDoKpiE/bMPHFW33yzQgaO5eu7Wy/TD1vUCqzBl3gZNmbdRP25XX1P4Rz6wxykRE1u26Wv95IXlkbPCK/6rN2eg4nQQdh84qsf+uU4zVm+P7KF+jYp6/PrW6tUy2FcP4sTh27AEQAKgazERAF0Fk7T/5Hc36PfzN6ppnSqae1+PwL6P087+2VnAcWmtIu8NZUOgQOBITm7k9oDDx/I0975wLnAe5KNpS7v8PeNzjX17neyXRltIflDnJhrRr4WqVkgN8tRDMTcCIAHQtVAJgK6CSdrf7q3rPmlhZHHlJ25sG7mHJ2ibLQDddtxc7T+co38M7cqyEkE7QAGYT8FrDsdce7Fu5w0xvh2RrN0HNWrmGi1avysyZou61TTxP1rr0kZn+baPZB+IAEgAdP0OEABdBZO4/9TFmzRh9sc676xKWnB/z8AtsGyLV/d9Ml2VypVV5pi+LC2RxLVa2Ed/btEnmvTOevW5uK5euK09Qo4C9lT1y//aoifmrtfBo7kqXzZFQ69sprt7NA3czwfHj5rw7gRAAqBrERIAXQWTuL890df9twtlr296LK2VfhqwS6yvr9iqh2ZkqtMFZ+v1u1gAOolLtdCPnrFtj9KeXapqFVP1waN9+CXBoUg++mJf5PtmprZdfv7ZmjCgtZrWqeowKl0LEyAAEgDtTeYjJdnLWe1O/OslzYzhK0MAjAGLpqcKTF+2RY++tTZyI336yF6Bem3TA3/J0BvvZ+menk31wNUtOXwInCJgZ6wuHTdX+w7naOaQLmrXMDyvOAzK4Tx8LFdPL9ioqYs3RxaJtzBtC67f3KFhoFcICIpfcedBACQA9rc1NCWtkjSDAFjcrxL9iitwNCcv8naQrN2H9HD/lpFLPUHZev9ukTbtOqA/Dmqv3hfVDcq0mEfABO565X3NXbdDI/u10JBezQI2u2BPZ/nmr/XwjEx9+tWByESvbnWuxqa1Ut3qFYM98VIwOwIgAfDEMs4nAJaCb3UIP8JfVmVpxJsZqlm5XOS1WtUrlkv4p9hz8KjajXs3Mo/Vj/bR2VXKJ3xOTCCYAnbP2ui/r1XXZrX16p0dgznJgM1q76Fjmjj7o8grFm07p1oFjUu7RFdfwpt2SupQEQAJgLEGQHsNwomvQqgmKWvv3r2qXt2uBrMhELuAXUbrNyVdn+z8Rvf2bh55n2eit4Xrd+qOaSt1fu0qWjiiZ6Knw/4DLLBxx371eTI98pDCDwkwno7U0k1fR+79te3Wjo30YP+WgfjFz9PkS0kjAiABMNYAOEbS6JPrnwBYSn4iJPBjzMr8Qve8tlpVypfVkgevTPgZt8lz1+v3Cz7RgMsaaPLAdgmUYddBF7D16rr+ZqG27/n2DRVs3gQuqFNFEwe0iTzswVbyAgRAAmCsAZAzgCX/PU2KPdqae9c9+57WbN+nn3c7X4/86OKEfu5bX1yupZ98rcevv0S3dmyc0Lmw8+AL2JJBSzZ+JQuDbEULnFW5vH7Upp4qlitbdGNaxEWAAEgAjDUAnlyIPAUcl69mcg5acNm1QmqKFo/spXNrJOZGcLsk3WbMnMir6mb/qpsuqsftDclZkXxqBEqvAAGQAEgALL3f79B9Mjt7MnDqMq3csjtyX5C97zMRm61H1v+pJZHXTWWM7hvY19QlwoZ9IoBA6RAgABIAbYXNgnULPpA0XNJCSdmStnooc84AekCiiXeBFZ9mR0JgakqZyNtBGtWq7L2zTy1f+/dneuRva9SlWS29dmcnn0ZlGAQQQCA4AgRAAqA93miB7+TtZUm3eyhVAqAHJJrEJnDbSyuUvmGXBlzaQJNvKvkHMIa/8aFmrN6ue69spuF9W8Q2eVojgAACIRAgABIAXcuUAOgqSP9TBDKz9uraZ95TmTLSnGHddWFdW22o5LZeTyyKLEw77Y4O6tXinJLbMXtCAAEESkiAAEgAdC01AqCrIP1PKzB4+iq9s/bLyJsB/vBTe1NhyWzZB47qsse+XQA649d9VaNy4helLplPzl4QQCCZBAiABEDXeicAugrS/7QCtrhu3ynpslU13hrSRW1L6B2r89bt0J2vvK+mdapo/v0sAE15IoBA6RQgABIAXSubAOgqSP9CBQruxevWvLam/6xkXrE16Z2P9dyiTRrY/jxNuqEtRwcBBBAolQIEQAKga2ETAF0F6V+owNavD+rK3y1STl6+Xr+rkzpdUCvuWjc/v0zLN2dr4oDWuvnyRnHfHztAAAEEEiFAACQAutYdAdBVkP5nFBg1M1OvLt+q9o3P0puDO6uMPRkSpy0nN0+tx8zVoWO5mntfyT98EqePxbAIIIDAKQIEQAKg69eCAOgqSP8zCuzYd1jdJy3UkZy8uD+Vu2b7Xl3z9HuqVjE18gBISkr8wiaHHQEEEEikAAGQAOhafwRAV0H6FykwftZHej59s1rVr663f9k1bsHslWVb9Ou31qr7hXX0yn9eXuS8aIAAAgiEVYAASAB0rV0CoKsg/YsUsKVZ7CzgN0dy9Owtl0VeIh+PbdjrH2jmh59r2FXNNeyqC+OxC8ZEAAEEAiFAACQAuhYiAdBVkP6eBJ58d4Oemr8xsjyLLQ6dWjbFU79YGlnI3Jp9MHL2z84CsiGAAAKlVYAASAB0rW0CoKsg/T0J7D98TN0mLdSeg8f02xva6Mb2DT3189po1/4j6vD4vMjbRzJG91X1iiwA7dWOdgggED4BAiAB0LVqCYCugvT3LPB8+iaNn/WxGtSspAUjeqhCalnPfYtqOGftl7p7+iq1qFtNc+7rXlRz/h4BBBAItQABkADoWsAEQFdB+nsWOHwsN3Iv4M79RzQurZVu69zEc9+iGk6Y/ZGmLt6sn1zeUBMGtCmqOX+PAAIIhFqAAEgAdC1gAqD91sCNAAAQpElEQVSrIP1jEpi+/DM9OnON6lSroPSRvVSpvD9nAQf+YZlWbMnWpBvaaKDPl5dj+oA0RgABBEpAgABIAHQtMwKgqyD9YxI4mpMXeTtI1u5Deqh/Sw3u0TSm/qdrbGO2HjMnstbgvOE91Oycqs5jMgACCCAQZAECIAHQtT4JgK6C9I9Z4K+rsnT/mxmqUamcljzYy/mBjYxte5T27FLVrFxOq0f1ids6gzF/UDoggAACcRIgABIAXUuLAOgqSP+YBXLz8tVvSro+2fmN7u3dXMP7uK3ZN23ppxr79jr1alFH0+5gAeiYDwgdEEAgdAIEQAKga9ESAF0F6V8sgdmZX+gXr61WlfJllf5AL9WqWqFY41inoX/+QG9nfK77+1yoob2bF3scOiKAAAJhESAAEgBda5UA6CpI/2IJ5Ofn67pnlipz+17d2fV8jbrm4mKNY526TFyg7XsO6U93dtQVzWoXexw6IoAAAmERIAASAF1rlQDoKkj/YgssWr9Tt09bqfKpKVo8sqfq1agU81g79h1Wx/HzlVJGyhzTT1UqpMY8Bh0QQACBsAkQAAmArjVLAHQVpH+xBews4E1Tl0eWb7mlYyONv751zGMVXEq+qF51zf5Vt5j70wEBBBAIowABkADoWrcEQFdB+jsJrPg0WwOnLlNqShnNv7+HGteqEtN4j/9znV5Y8qlu7dhIjxcjQMa0MxojgAACAREgABIAXUuRAOgqSH9ngUEvrdDiDbt0/aUN9ORN7WIab8BzS7V66x5NHthWAy47L6a+NEYAAQTCKkAAJAC61i4B0FWQ/s4CmVl7de0z76lMGemdX3VXi3OreRrzSE6uWo+eq6O5eVo0oqea1I7t7KGnndAIAQQQCKAAAZAA6FqWBEBXQfr7IvCLV1dp9pov1a9VXU39aXtPY67eulsDnvuXzq5SXqtGXaUyliDZEEAAgSQQIAASAF3LnADoKkh/XwQ27tgfWRw6L196a0gXtW1Ys8hxX1yyWf/9z4901UXn6MVBHYpsTwMEEECgtAgQAAmArrVMAHQVpL9vAve/kaG/rs5St+a1Nf1nHYsc957XVmlW5pd64OoWuqdnsyLb0wABBBAoLQIEQAKgay0TAF0F6e+bwLbsg7ryd4t0LDdff/55J3VuWuuMY3caP19f7jus1+/qpE4XnLmtb5NkIAQQQCAAAgRAAqBrGRIAXQXp76vAozPXaPryz/SDxmfpL4M7F3pf3+d7DumKiQtUNqWMMsf0VeXyLADt64FgMAQQCLQAAZAA6FqgBEBXQfr7KmBv9ug+aaGO5ORp2u0d1KvlOacd3979a+8AvqRBdf1jKAtA+3oQGAwBBAIvQAAkALoWKQHQVZD+vgtMmPWRpqZv1sX1LNx1VYq95+2kbezbazVt6RYN6txYY9Mu8X0ODIgAAggEWYAASAB0rU8CoKsg/X0X2H3gqLpNWqhvjuTo2Vsu04/a1DtlH2nPLlXGtj166uZ2SmvXwPc5MCACCCAQZAECIAHQtT4JgK6C9I+LwJR5GzRl3kZdUKeK5g7rrtSyKd/t5/CxXLUeMyfysMiSB3qp4dmV4zIHBkUAAQSCKkAAJAC61iYB0FWQ/nER2H/4WORewN0Hj2nSDW00sH3D7/azcku2bvzDMtWuWkErH+nNAtBxOQIMigACQRYgABIAXeuTAOgqSP+4CTyfvknjZ32sBjUracGIHqqQWjayr6mLN2nC7I9jemtI3CbJwAgggEACBAiABEDXsiMAugrSP24Cdqm3x28Xase+Ixp7XSsNuqJJZF93T39fc9bu0MP9W+ruHk3jtn8GRgABBIIqQAAkALrWJgHQVZD+cRV4dflnGjVzTeRyb/oDPVWpXFldPn6+du0/ojcHd1aHJmfHdf8MjgACCARRgABIAHStSwKgqyD94ypwNCdPvScv0rbsQ3rw6pa6pk29yBPCqSlltGZsP1Us9+1lYTYEEEAgmQQIgARAq/chkkZKOldShqShklZ4/CIQAD1C0SxxAjNWZ2n4GxmqUamcRvRrIXtbSNuGNfXWkC6JmxR7RgABBBIoQAAkAN4k6RVJgyX9W9IwSTdKaiFpp4faJAB6QKJJYgVy8/J19ZR0bdz5japWSI2sD3hHlyYafW2rxE6MvSOAAAIJEiAAEgAt9K2U9MtoDdpiadskPS1pooe6JAB6QKJJ4gXeWfOFBr+6+ruJPP2TS3Vt2/qJnxgzQAABBBIgQABM7gBYXtJBSTdImnlC/b0sqaaktNPUZAVJ9l/BVk1S1t69e1W9umVBNgSCKZCfn6/rnlmqzO17IxNc+tCVkeVh2BBAAIFkFCAAJncAtNMf2yVdIWnZCV+ASZJ6SOp4mi/FGEmjT/5zAmAy/vgI32dO37BLt720Qo1rVdaiET1ZADp8h5AZI4CATwIEQAJgrAGQM4A+ffkYJjEC7238SufWqKhm51RNzATYKwIIIBAAAQJgcgfA4lwCPrlsuQcwAF9kpoAAAggggEAsAgTA5A6AViv2EIgt+WJLv9hmD4FslfQMD4HE8lWiLQIIIIAAAuERIAASAG0ZGHvo4+5oELRlYAZKailph4dS5gygBySaIIAAAgggECQBAiAB0OrRloApWAj6Q0n3Rs8MeqlVAqAXJdoggAACCCAQIAECIAHQtRwJgK6C9EcAAQQQQKCEBQiABEDXkiMAugrSHwEEEEAAgRIWIAASAF1LjgDoKkh/BBBAAAEESliAAEgAdC05AqCrIP0RQAABBBAoYQECIAHQteQIgK6C9EcAAQQQQKCEBQiABEDXkiMAugrSHwEEEEAAgRIWIAASAF1LjgDoKkh/BBBAAAEESliAAEgAdC05AqCrIP0RQAABBBAoYQECIAHQteQIgK6C9EcAAQQQQKCEBQiABEDXkiMAugrSHwEEEEAAgRIWIAASAF1LLhIAt23bpurV7X/ZEEAAAQQQQCDoAhYAGzZsaNOsIWlf0Ocbj/mVicegSTRmA0lZSfR5+agIIIAAAgiUJoHzJG0vTR/I62chAHqVOn0786svab/bMKftXS0aLq044zF+HKacsCGx8k6PFVbeBby3pK6w8i7gvWW868rG/1xSvvcplZ6WBMDgHsvI5eVkPj0dw6HByjsWVlh5F/DekrrCyruA95bUlXermFsSAGMmK7EOFL53aqyw8i7gvSV1hZV3Ae8tqSusvAvEsSUBMI64jkPzQ8I7IFZYeRfw3pK6wsq7gPeW1BVW3gXi2JIAGEdcx6ErSHpY0gRJRxzHKu3dsfJ+hLHCyruA95bUFVbeBby3pK68W8XckgAYMxkdEEAAAQQQQACBcAsQAMN9/Jg9AggggAACCCAQswABMGYyOiCAAAIIIIAAAuEWIACG+/gxewQQQAABBBBAIGYBAmDMZHRAAAEEEEAAAQTCLUAADObxGyJppKRzJWVIGippRTCnmtBZdY86/UBSPUnXS5qZ0BkFc+f2NPkASS0lHZL0L0kPSlofzOkmfFa/kGT/NYnOZK2kcZJmJ3xmwZ7AQ9FVC56SNCzYU03I7MZIGn3Snu07aN9LtlMF7FWrv5HUX1JlSZ9IukPS+2D5I0AA9MfRz1FukvSKpMGS/h39QXqjpBaSdvq5o1Iwlv1g6CJplaQZBMBCj+g7kl6XtFJSqqTxki6RdLGkA6WgDvz+CNdKypW0UZL9jBwU/UXjUkkWBtlOFegg6Q1J+yQtJACetkQsAN4g6aoT/jZH0lcU1CkCZ0n6IFpL/yNpl6TmkjZF/4PMBwECoA+IPg9hoc/+of5ldNwUSdskPS1pos/7Kk3D2bscOQPo7YjWif4y0UNSurcuSd8qOxoC/5j0EqcCVJW0WtI9kkZJ+pAAWGgA/LGkdtRQkQL2b539ct+tyJY0KLYAAbDYdHHpWF7SwehviSdeynxZUk1JaXHZa+kYlADo/Tg2i57dai1pjfduSdmyrCQ7A2/fQTsDuC4pFc78oc3GAvJ9khYRAAvFsjOAdmuPveP9sKRl0cX+t1JTpwjY92yOpPMk2S+q2yU9J+kFrPwTIAD6Z+nHSPWjhX5F9IdDwZiTol+Cjn7spJSOQQD0dmDtjPLfo79QdPXWJSlbWTi2f6ArSvpG0i2SZiWlxJk/9M2SHpFkl4At1BAAC/eyW1bsbKnd92f3LNv9gHafm92OsZ/a+p6A1ZJtkyW9Ga0vu7fUbo2yXzjYfBAgAPqA6OMQBMDiYxIAvdnZ/TT2D5GFvyxvXZKylZ2NbySpRvSM/J3RX8I4A3i8HBpGb8jvI+n/on9MAPT+dbGrOp9JGi6JWwu+73Y0Wlt2MqRg+300CHb2TkzLMwkQAINVH1wCLv7xIAAWbfdM9DYCe3r606Kb0+IEgXnRm8/vRuU7Abuf7W/RB2YK/tAumdt3MU+SvcfVHqZhK1zA7ve22rIn9dmOC1gwfleS/eJVsNmT+XaPqZ01ZfNBgADoA6LPQ9hDILbkiy39YptdsrN7ROwfbx4CKRybAFi4jX3P7SEie0imZ/T+P5/LttQPtyD6Pby91H9S7x+wmqTGJzWfJunj6PId3F96Zku7HGw/2+3eQDu7xXZc4E+S7AzziQ+BPCnJboM68awgZg4CBEAHvDh1tWVg7B4HO9NgQdDW0xoYXStqR5z2GdZh7QeoPdBgmy0ZYJdSbAkKuyGdG6uPH1W7edruYbOHiE5c+89uRrd1Adm+LzAhuuaf1ZCFHLOzdRP7Rc9K4FW4AJeAC7d5QtLb0cu+drvP2OgTwbYcky1zwnZcwO4ptfVK7T5JW17o8ugDIHdJeg0ofwQIgP44+j2KLQFTsBC0Lalwb3RNQL/3E/bx7GyWBb6TNwvQnKk5rmJnR0+32aKq/xv2IojD/O1+rN7RG/UtJNv9bbYgrV2SYjuzAAGwcB9bi9Nuv6gVDXzvRR+gsbXt2E4VuCa6sLit/2e3rNgDITwF7GOlEAB9xGQoBBBAAAEEEEAgDAIEwDAcJeaIAAIIIIAAAgj4KEAA9BGToRBAAAEEEEAAgTAIEADDcJSYIwIIIIAAAggg4KMAAdBHTIZCAAEEEEAAAQTCIEAADMNRYo4IIIAAAggggICPAgRAHzEZCgEEEEAAAQQQCIMAATAMR4k5IoAAAggggAACPgoQAH3EZCgEEEAAAQQQQCAMAgTAMBwl5ogAAggggAACCPgoQAD0EZOhEEAAAQQQQACBMAgQAMNwlJgjAggggAACCCDgowAB0EdMhkIAAQQQQAABBMIgQAAMw1FijggggAACCCCAgI8CBEAfMRkKAQQQQAABBBAIgwABMAxHiTkigAACCCCAAAI+ChAAfcRkKAQQQAABBBBAIAwCBMAwHCXmiAACCCCAAAII+ChAAPQRk6EQQAABBBBAAIEwCBAAw3CUmCMCCCCAAAIIIOCjAAHQR0yGQgABBBBAAAEEwiBAAAzDUWKOCCCAAAIIIICAjwIEQB8xGQoBBBBAAAEEEAiDAAEwDEeJOSKAAAIIIIAAAj4KEAB9xGQoBBBAAAEEEEAgDAIEwDAcJeaIAAIIIIAAAgj4KEAA9BGToRBAAAEEEEAAgTAIEADDcJSYIwIIIIAAAggg4KMAAdBHTIZCAAEEEEAAAQTCIEAADMNRYo4IIIAAAggggICPAgRAHzEZCgEEEEAAAQQQCIMAATAMR4k5IoAAAggggAACPgr8P+LaHQzYEH4vAAAAAElFTkSuQmCC" width="640">





    [<matplotlib.lines.Line2D at 0x7f6601667278>]


