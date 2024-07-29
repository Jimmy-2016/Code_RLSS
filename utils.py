


import sys
import numpy as np
import matplotlib.pyplot as plt
import random
# from scipy.optimize import curve_fit
import matplotlib
import scipy.stats
import seaborn as sns
import pickle

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)


'''
This file provides basic parameters and function needed for the simulation of the model
'''


##  Params
random.seed(1)  # this two freeze the random seed for reproducibility purposes
np.random.seed(1)

epsilon = 0.0  # in the case of e_greedy policy, set this to the level of desired epsilon
esp_decay = 0.99  # decay of epsilon in each iteration
lr = .1  # learning rate
Gamma = .9  # discount factor
K = .4  # drift coeff
SoftOrGreed = 1  # choose between explorations methods (1 = softmax, 0= e_greedy)
beta = 50  # beta in softmax
Bin = 100  # this is used for smoothing and visualization purposes, it should be used based on trial numbers
SigmaEv = 1  # variance of stimulus
M = 100  # M determines the range of states
Delta = 3  # determines the resolution of the state space
delta_t = 1  # delta t
# States = np.linspace(-M, M, 2 * int(M/Delta) + 1)
States = np.linspace(-M, M, 2 * M * Delta + 1)  # state space (S)

mid_state = np.floor(len(States)/2)
Numstates = len(States)
Actions = [-1, 1, 0]  # actions left, right, wait respectively
TerminatingActions = [-1, 1]
Reward = [20, -50, -1]  # True, False, Wait
NumActions = len(Actions)
Q = 0*np.ones((Numstates, NumActions))  # Q-table
NumTr = 90  # Number of trails for simulations
Coherence = np.array([0, 3.2, 6.4, 12.8, 25.6, 51.2])/100  # coherence level used
DiVector = np.hstack((np.ones(int(NumTr/2)), -1*np.ones(int(NumTr/2))))  # half of the direction of rigth (+) and the other half is left (-1)
random.shuffle(DiVector) #  randomize order of directions
AllCoh = np.repeat(Coherence, NumTr/len(Coherence))  # create same number of trials for each coherence
random.shuffle(AllCoh)  # randomize order of coherence

NumIter = 2  # Number of simulations (N)

##

def MySmooth(X, Bin):

    '''
    This function smooth any signal
    :param X: signal
    :param Bin: smooth windows size
    :return: smoothes signal
    '''

    X = np.hstack((X[0]*np.ones(Bin), X, X[-1]*np.ones(Bin)))
    X = np.convolve(X, np.ones(Bin)/Bin, mode='same')

    X = np.delete(X, np.arange(Bin))
    X = np.delete(X, np.arange(X.shape[0]-Bin, X.shape[0]))
    return X


def find_nearest(array, value):
    '''
    This function finds the nearest neighbor in S
    :param array: state array
    :param value: the value that its nearest neighbor is to be found
    :return: nearest neighbor
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def softmax(x, beta):
    '''
    #  implementation of softamx function
    :param x: the value that its softmax inference is required
    :param beta: control the niose level of softmax
    :return: entry and value that has been chosen by the softmax function
    '''
    e_x = np.exp(x * beta)
    Z = e_x / e_x.sum()
    rndnum = random.uniform(0, 1)
    try:
        indx = np.where(rndnum < np.hstack((0, np.nancumsum(Z))))[0][0] - 1
    except:
        indx = np.argmax(x)
    return np.hstack((indx, Z))


def TakeAction(st,epsilon, ev, Dtemp, Q):
    '''
    # taking action in our model
    :param st: current state
    :param epsilon: in the case that e_greedy is chosen, this parameter controls the level of exploration
    :param ev: evident sampeled form the N(K*Dir*Coh, Sigma)
    :param Dtemp: true direction of the stimuli
    :param Q: Q table
    :return: new State, chosen action, assigned reward
    '''
    if sum(np.diff(Q[st, :])) == 0:
        NextAction = Actions[random.randrange(0, len(Actions))]
    else:
        if SoftOrGreed:   # softmax
            tmpq = Q[st, :]
            NextAction = Actions[softmax(tmpq, beta=beta)[0].astype(int)]
        else:   # e_greedy
            if np.random.rand() < epsilon:
                NextAction = Actions[random.randrange(0, len(Actions))]  # Explore
            else:
                NextAction = Actions[np.argmax(Q[st, :])]  # Exploit
            # NextAction = Actions[np.argmax(Q[st, :])]

    if NextAction == 0:  # state transition
        # NexState = np.round(st + ev)
        NexState = int(find_nearest(States, delta_t * (States[st] + ev)))
    else:
        NexState = st

    FedBack = 2
    if NextAction == 1 or NextAction == -1:
        if NextAction == Dtemp:  # correct
            FedBack = 0
        else:
            FedBack = 1   # incorrect
    R = Reward[FedBack]
    A = Actions.index(NextAction)
    S = int(NexState)

    return S, A, R

