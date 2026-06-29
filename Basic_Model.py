"""
Basic_Model.py — run the RL-DDM simulation.

This is the simulation engine for the RL-DDM (reinforcement-learning drift-diffusion
model). It trains the agent on the trial sequence and parameters defined in utils.py:
on every trial the agent accumulates noisy evidence step by step (the "wait" action),
updates its Q-table via temporal-difference learning, and eventually commits to a
left/right choice. The decision threshold therefore emerges from learning rather than
being imposed.

What it does:
  - repeats the whole training run NumIter times (independent agents),
  - records, for every iteration, the full Q-table over training (AllQ) and the
    terminal/boundary state reached on each trial (AllBound).

Inputs : all model parameters and the trial sequence come from utils.py.
Output : SavedData/ModelSim.pkl   ->  pickle of [AllBound, AllQ].

A directory named SavedData/ must exist in the repo root before running.
Run:  python Basic_Model.py
Then: python Plot_Basic_Model.py   (reads ModelSim.pkl and reproduces the figures).
"""

from utils import *



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


