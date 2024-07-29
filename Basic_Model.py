
from utils import *


'''
This file runs the models based on the params set in utils
it saved the simulation in SavedData folder in the root directory
'''



AllQ = np.zeros((NumIter, len(AllCoh), Numstates, NumActions))
AllBound = np.zeros((NumIter, NumTr))

for iti in range(NumIter):
    print(iti)
    Q = 0 * np.ones((Numstates, NumActions))  # Q-table
    trCount = 0
    st = int(np.floor(len(States) / 2))
    reset = 1
    Bound = np.zeros(len(AllCoh))
    while True:
        if reset:
            Ctemp = AllCoh[trCount]
            Dtemp = DiVector[trCount]
        reset = 0

        epsilon = epsilon * esp_decay

        ev = np.random.normal(K*Dtemp*Ctemp, SigmaEv)  # takes a sample

        NexState, NextAction, R = TakeAction(st, epsilon, ev, Dtemp, Q)  # take an action

        if NextAction == 0 or NextAction == 1:  # update the Q_table
            Q[st, NextAction] = Q[st, NextAction] + lr * (R - Q[st, NextAction])
        else:
            Q[st, NextAction] = Q[st, NextAction] + lr*(R + Gamma*np.max(Q[NexState, :]) - Q[st, NextAction])

        if NextAction == 0 or NextAction == 1:
            AllQ[iti, trCount, :] = Q
            reset = 1
            Bound[trCount] = NexState
            trCount = trCount + 1
            st = int(np.floor(len(States) / 2))  # set to zero
        else:
            st = NexState
        if trCount >= len(AllCoh):
            break

    AllQ[iti] = Q
    AllBound[iti] = Bound


with open('./SavedData/ModelSim.pkl', 'wb') as f:
    pickle.dump([AllBound, AllQ], f)


