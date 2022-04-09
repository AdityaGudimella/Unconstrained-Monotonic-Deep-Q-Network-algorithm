# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import random
from collections import deque
import numpy as np
from torch.utils.data import Dataset
from morl.memories import AtariExperienceReplay
from morl import experiences as exp


###############################################################################
############################### Class ReplayMemory ############################
###############################################################################


class ReplayMemory(Dataset):
    """
    GOAL: Implementing the replay memory required for the Experience Replay
          mechanism of the DQN Reinforcement Learning algorithm. This class
          inherits from the Dataset class from Pytorch for being used with
          efficient data loaders.

    VARIABLES:  - memory: Data structure storing the RL experiences.

    METHODS:    - __init__: Initialization of the memory data structure.
                - __getitem__: Get an item from the replay memory.
                - __len__: Return the length of the replay memory.
                - push: Insert a new experience into the replay memory.
                - sample: Sample a batch of experiences from the replay memory.
                - reset: Reset the replay memory.
    """

    def __init__(self, capacity=10000):
        """
        GOAL: Initialization of the replay memory data structure.

        INPUTS: - capacity: Capacity of the data structure, specifying the
                            maximum number of experiences to be stored
                            simultaneously into the data structure.

        OUTPUTS: /
        """

        random.seed(0)
        self.capacity = capacity
        # self.memory = deque(maxlen=capacity)
        self.memory = AtariExperienceReplay(
            batch_size=32,
            capacity=capacity,
            frame_stack=4,
            state_height=84,
            state_width=84,
            n=1,
        )

    def __getitem__(self, index):
        """
        GOAL: Outputing the item associated with the provided index
              from the replay memory.

        INPUTS: /

        OUTPUTS: - item: Selected item of the replay memory.
        """

        return self.memory[index]

    def __len__(self):
        """
        GOAL: Return the size of the replay memory, i.e. the number of experiences
              currently stored into the data structure.

        INPUTS: /

        OUTPUTS: - length: Size of the replay memory.
        """

        return len(self.memory)

    def push(self, state, action, reward, nextState, done):
        """
        GOAL: Insert a new experience into the replay memory. An experience
              is composed of a state, an action, a reward, a next state and
              a termination signal.

        INPUTS: - state: RL state of the experience to be stored.
                - action: RL action of the experience to be stored.
                - reward: RL reward of the experience to be stored.
                - nextState: RL next state of the experience to be stored.
                - done: RL termination signal of the experience to be stored.

        OUTPUTS: /
        """

        # FIFO policy
        # self.memory.append((state, action, reward, nextState, done))
        exp_ = exp.Experience(
            episode_ids=[],
            states=state,
            actions=np.asarray([[action]], dtype=np.uint8),
            rewards=np.asarray([[reward]], dtype=np.float32),
            next_states=nextState,
            dones=np.asarray([[done]]),
            timestamps=[],
        )
        self.memory.add_(exp_)
        print(len(self.memory))

    def sample(self, batchSize):
        """
        GOAL: Sample a batch of experiences from the replay memory.

        INPUTS: - batchSize: Size of the batch to sample.

        OUTPUTS: - state: RL states of the experience batch sampled.
                 - action: RL actions of the experience batch sampled.
                 - reward: RL rewards of the experience batch sampled.
                 - nextState: RL next states of the experience batch sampled.
                 - done: RL termination signals of the experience batch sampled.
        """

        # state, action, reward, nextState, done = zip(
        #     *random.sample(self.memory, batchSize)
        # )
        # return state, action, reward, nextState, done
        assert batchSize == 32
        exp_ = self.memory.get_().as_default_tensor()
        return exp_.states, exp_.actions.squeeze(1), exp_.rewards.squeeze(1), exp_.next_states, exp_.dones.squeeze(1)

    def reset(self):
        """
        GOAL: Reset (empty) the replay memory.

        INPUTS: /

        OUTPUTS: /
        """

        # random.seed(0)
        # self.memory = deque(maxlen=self.capacity)
        raise NotImplementedError()
