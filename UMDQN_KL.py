# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import math

import numpy as np
import pandas as pd
import scipy.stats as stats

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# pylint: disable=E1101
# pylint: disable=E1102

from replayMemory import ReplayMemory

from Models.UMDQN_KL_Model import UMDQN_KL_Model
from Models.UMDQN_KL_Model_Atari import UMDQN_KL_Model_Atari

from DQN import DQN



###############################################################################
################################ Class UMDQN_KL ###############################
###############################################################################

class UMDQN_KL(DQN):
    """
    GOAL: Implementing the UMDQN_KL Deep Reinforcement Learning algorithm.
    
    VARIABLES: - device: Hardware specification (CPU or GPU).
               - gamma: Discount factor of the RL algorithm.
               - learningRate: Learning rate of the DL optimizer (ADAM).
               - epsilon: Epsilon value for the DL optimizer (ADAM).
               - targetNetworkUpdate: Update frequency of the target network.
               - learningUpdatePeriod: Frequency of the learning procedure.
               - batchSize: Size of the batch to sample from the replay memory.
               - capacity: Capacity of the replay memory.
               - replayMemory: Experience Replay memory.
               - rewardClipping: Clipping of the RL rewards.
               - gradientClipping: Clipping of the training loss.
               - optimizer: DL optimizer (ADAM).
               - epsilonStart: Initial value of epsilon (Epsilon-Greedy).
               - epsilonEnd: Final value of epsilon (Epsilon-Greedy).
               - epsilonDecay: Exponential decay of epsilon (Epsilon-Greedy).
               - epsilonTest: Test value of epsilon (Epsilon-Greedy).
               - epsilonValue: Current value of epsilon (Epsilon-Greedy).
               - policyNetwork: Deep Neural Network representing the info used by the RL policy.
               - targetNetwork: Deep Neural Network representing the target network.
                                
    METHODS: - __init__: Initialization of the RL algorithm.
             - chooseAction: Choose a valid action based on the current state
                             observed, according to the RL policy learned.
             - learning: Execute the RL algorithm learning procedure.
    """

    def __init__(self, observationSpace, actionSpace, environment,
                 parametersFileName='', reporting=True):
        """
        GOAL: Initializing the RL agent based on the UMDQN_KL Deep Reinforcement Learning
              algorithm, by setting up the algorithm parameters as well as 
              the Deep Neural Networks.
        
        INPUTS: - observationSpace: RL observation space.
                - actionSpace: RL action space.
                - environment: Name of the RL environment.
                - parametersFileName: Name of the JSON parameters file.
                - reporting: Enable the reporting of the results.
        
        OUTPUTS: /
        """

        # Initialization of the DQN parent class
        DQN.__init__(self, observationSpace, actionSpace, environment, parametersFileName, False)

        # Setting of the parameters
        if parametersFileName == '':
            parametersFileName = ''.join(['Parameters/parameters_UMDQN_KL_', str(environment), '.json'])
        parameters = self.readParameters(parametersFileName)

        # Set the device for DNN computations (CPU or GPU)
        self.device = torch.device('cuda:'+str(parameters['GPUNumber']) if torch.cuda.is_available() else 'cpu')

        # Set the general parameters of the RL algorithm
        self.gamma = parameters['gamma']
        self.learningRate = parameters['learningRate']
        self.epsilon = parameters['epsilon']
        self.targetNetworkUpdate = parameters['targetNetworkUpdate']
        self.learningUpdatePeriod = parameters['learningUpdatePeriod']
        self.rewardClipping = parameters['rewardClipping']
        self.gradientClipping = parameters['gradientClipping']

        # Set the Experience Replay mechanism
        self.batchSize = parameters['batchSize']
        self.capacity = parameters['capacity']
        self.replayMemory = ReplayMemory(self.capacity)

        # Set the distribution support
        self.numberOfAtoms = parameters['numberOfAtoms']
        self.minReturn = parameters['minReturn']
        self.maxReturn = parameters['maxReturn']
        self.support = np.linspace(self.minReturn, self.maxReturn, self.numberOfAtoms)
        self.supportTorch = torch.linspace(self.minReturn, self.maxReturn, self.numberOfAtoms, device=self.device)
        self.supportRepeatedBatchSize = self.supportTorch.repeat(self.batchSize, 1).view(-1, 1)

        # Initialization of the variables required for the faster but less accurate computation of expectation
        self.fastExpectation = parameters['fastExpectation']
        self.numberOfPoints = parameters['numberOfPoints']
        self.uniformProba = 1/(self.maxReturn - self.minReturn)
        self.delta = self.support[1] - self.support[0]

        # Set the two Deep Neural Networks of the RL algorithm (policy and target)
        self.atari = parameters['atari']
        if self.atari:
            self.policyNetwork = UMDQN_KL_Model_Atari(observationSpace, actionSpace, parameters['structureUMNN'], parameters['stateEmbedding'], parameters['numberOfSteps'], self.device).to(self.device)
            self.targetNetwork = UMDQN_KL_Model_Atari(observationSpace, actionSpace, parameters['structureUMNN'], parameters['stateEmbedding'], parameters['numberOfSteps'], self.device).to(self.device)
        else:
            self.policyNetwork = UMDQN_KL_Model(observationSpace, actionSpace, parameters['structureDNN'], parameters['structureUMNN'], parameters['stateEmbedding'], parameters['numberOfSteps'], self.device).to(self.device)
            self.targetNetwork = UMDQN_KL_Model(observationSpace, actionSpace, parameters['structureDNN'], parameters['structureUMNN'], parameters['stateEmbedding'], parameters['numberOfSteps'], self.device).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

        # Set the Deep Learning optimizer
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=self.learningRate, eps=self.epsilon)

        # Set the Epsilon-Greedy exploration technique
        self.epsilonStart = parameters['epsilonStart']
        self.epsilonEnd = parameters['epsilonEnd']
        self.epsilonDecay = parameters['epsilonDecay']
        self.epsilonTest = parameters['epsilonTest']
        self.epsilonValue = lambda iteration: self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-1 * iteration / self.epsilonDecay)

        # Initialization of the experiment folder and tensorboard writer
        self.initReporting(parameters, 'UMDQN_KL')


    def chooseAction(self, state, plot=False):
        """
        GOAL: Choose a valid RL action from the action space according to the
              RL policy as well as the current RL state observed.
        
        INPUTS: - state: RL state returned by the environment.
                - plot: Enable the plotting of the random returns distributions.
        
        OUTPUTS: - action: RL action chosen from the action space.
        """

        # Choose the best action based on the RL policy
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            if self.fastExpectation:
                expectedReturns = self.policyNetwork.getExpectation(state, self.minReturn, self.maxReturn, self.numberOfPoints).squeeze(0)
            else:
                pdfs = self.policyNetwork(state, self.supportTorch.unsqueeze(1))
                pdfs = torch.exp(pdfs)
                expectedReturns = (pdfs * self.supportTorch).sum(1)/(self.numberOfAtoms*self.uniformProba)
            _, action = expectedReturns.max(0)

        # If required, plot the return distribution associated with each action
        if plot:
            colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown']
            plt.figure()
            ax = plt.subplot(1, 1, 1)
            accurateSupport = np.linspace(self.minReturn, self.maxReturn, self.numberOfAtoms*10)
            accurateSupportTorch = torch.linspace(self.minReturn, self.maxReturn, self.numberOfAtoms*10, device=self.device)
            with torch.no_grad():
                pdfs = self.policyNetwork(state, accurateSupportTorch.unsqueeze(1))
                pdfs = torch.exp(pdfs)
                expectedReturns = ((pdfs * accurateSupportTorch).sum(1))/(self.numberOfAtoms*10*self.uniformProba)
            for a in range(self.actionSpace):
                pdf = pdfs[a]
                expectedReturn = expectedReturns[a]
                ax.plot(accurateSupport, pdf.cpu(), linestyle='-', label=''.join(['Action ', str(a)]), color=colors[a])
                #ax.fill_between(accurateSupport, accurateSupport*0, pdf.cpu(), alpha=0.25, color=colors[a])
                #ax.axvline(x=expectedReturn, linewidth=2, linestyle='--', label=''.join(['Action ', str(a), ' expected return']), color=colors[a])
            ax.set_xlabel('Random return')
            ax.set_ylabel('PDF')
            ax.set(xlim=(-0.5, 1.5), ylim=(0, 3.5))
            ax.legend()
            plt.show()
            """
            plt.savefig("Figures/Distributions/UMDQN_KL.pdf", format='pdf')
            # Saving of the data into external files
            dataPDF = {
            'Action0_x': accurateSupport,
            'Action0_y': pdfs[0].cpu(),
            'Action1_x': accurateSupport,
            'Action1_y': pdfs[1].cpu(),
            'Action2_x': accurateSupport,
            'Action2_y': pdfs[2].cpu(),
            'Action3_x': accurateSupport,
            'Action3_y': pdfs[3].cpu(),
            }
            dataframePDF = pd.DataFrame(dataPDF)
            dataframePDF.to_csv('Figures/Distributions/UMDQN_KL.csv')
            quit()
            """
        
        return action.item()


    def learning(self):
        """
        GOAL: Sample a batch of past experiences and learn from it
              by updating the Reinforcement Learning policy.
        
        INPUTS: /
        
        OUTPUTS: - loss: Loss of the learning procedure.
        """
        
        # Check that the replay memory is filled enough
        if (len(self.replayMemory) >= self.batchSize):
            
            # Sample a batch of experiences from the replay memory
            batch = self.dataLoaderIter.next()
            state = batch[0].float().to(self.device)
            action = batch[1].float().to(self.device)
            reward = batch[2].float().to(self.device)
            nextState = batch[3].float().to(self.device)
            done = batch[4].float().to(self.device)

            # Computation of the current return distribution, according to the policy DNN
            pdfs = self.policyNetwork(state, self.supportRepeatedBatchSize)
            selection = torch.tensor([self.actionSpace*i + action[i] for i in range(self.batchSize)], dtype=torch.long, device=self.device)
            pdfs = torch.index_select(pdfs, 0, selection).view(-1, 1)
            currentPdfs = torch.exp(pdfs)

            # Computation of the next action, according to the policy DNN
            with torch.no_grad():
                if self.fastExpectation:
                    expectedReturns = self.targetNetwork.getExpectation(nextState, self.minReturn, self.maxReturn, self.numberOfPoints)
                    #expectedReturns = self.policyNetwork.getExpectation(nextState, self.minReturn, self.maxReturn, self.numberOfPoints) # Double DQN improvement
                else:
                    pdfs = self.targetNetwork(nextState, self.supportRepeatedBatchSize)
                    #pdfs = self.policyNetwork(nextState, self.supportRepeatedBatchSize) # Double DQN improvement
                    pdfs = torch.exp(pdfs)
                    expectedReturns = (((pdfs * self.supportTorch).sum(1))/(self.numberOfAtoms*self.uniformProba)).view(-1, self.actionSpace)
                _, nextAction = expectedReturns.max(1)

            # Computation of the new distribution to be learnt by the policy DNN
            with torch.no_grad():
                r = reward.view(self.batchSize, 1).repeat(1, self.numberOfAtoms).view(-1, 1)
                support = (self.supportRepeatedBatchSize - r)/self.gamma
                targetPdfs = self.targetNetwork(nextState, support)
                selection = torch.tensor([self.actionSpace*i + nextAction[i] for i in range(self.batchSize)], dtype=torch.long, device=self.device)
                targetPdfs = torch.index_select(targetPdfs, 0, selection)
                targetPdfs = torch.exp(targetPdfs)/self.gamma
                for i in range(self.batchSize):
                    if done[i] == 1:
                        targetPdfs[i] = torch.tensor(stats.norm.pdf(self.support, reward[i].item(), self.delta)).to(self.device)
                targetPdfs = targetPdfs.clamp(min=0.00001)
                targetPdfs = targetPdfs.view(-1, 1)
            
            # Compute the loss
            loss = (targetPdfs*(targetPdfs.log()-currentPdfs.log())).sum()

            # Computation of the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # Perform the Deep Neural Network optimization
            self.optimizer.step()

            return loss.item()
