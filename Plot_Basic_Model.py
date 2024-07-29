
from utils import *


'''
This file plots the results of the simulations
it read the saved data in SavedData folder that has been created by Basic_Model file
'''


with open('./SavedData/ModelSim.pkl', 'rb') as f:
    AllBound, AllQ = pickle.load(f)

## parameter set

mid_state = np.floor(len(States)/2)
Q = AllQ.mean(0)
Bin = 1  # param for something the signals (here, is the threshold)
q_bin = 3  # param for something the q_table

##  plot Q-tabel
SoftQ = Q[-1, :]
print(Q)
fig = plt.figure()
ax = plt.axes()
ax.plot(States, MySmooth(SoftQ[:, 0], q_bin), label='Left', c='r', lw=4)
ax.plot(States, MySmooth(SoftQ[:, 1], q_bin), label='Right', c='b', lw=4)
ax.plot(States, MySmooth(SoftQ[:, 2], q_bin), label='Wait', c='g', lw=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim([-50, 50])
ax.set_ylim([-20, 5])
plt.legend(loc='best', frameon=False, facecolor=None)


## plot Q-tabel in different time points during training
# timbins = [0, 1000, 2000]
timbins = [0, 10, 20]

fig = plt.figure()
ax = plt.axes()
for timestep in range(len(timbins)):
    SoftQ = AllQ[:, timbins[timestep], :].mean(0)
    row = 1
    col = 3
    plt.subplot(row, col, timestep + 1)
    ax = plt.subplot(row, col, timestep + 1)

    # Hide the right and top spines
    if timestep == (col * (row - 1)):
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylim([-20, 6])

    else:
        ax.spines[['right', 'top', 'left', ]].set_visible(False)
        ax.get_yaxis().set_visible(False)

        # ax.axis("off")
        ax.set_ylim([-20, 6])

    if timestep == 0:
        SoftQ = np.zeros_like(SoftQ)

    plt.plot(States, MySmooth(SoftQ[:, 0], q_bin), label='Left', c='r', lw=4)
    plt.plot(States, MySmooth(SoftQ[:, 1], q_bin), label='Right', c='b', lw=4)
    plt.plot(States, MySmooth(SoftQ[:, 2], q_bin), label='Wait', c='g', lw=4)
    ax.set_xlim([-50, 50])
    ax.text(-25, 5, 'Trial Number: ' + str(timbins[timestep]), color='m',
                           bbox=dict(facecolor='none', edgecolor='m'))



# ### test the modes, with freezed q_table and plot Psychometric
st = int(np.floor(len(States) / 2))
Q = AllQ[:, -1, :].mean(0)
RT = np.zeros(NumTr)
Choice = np.zeros(NumTr)
TrajStates = len(AllCoh) * [[mid_state]]  # As a Starting Point
for i in range(len(AllCoh)):
    Ctemp = AllCoh[i]
    Dtemp = DiVector[i]
    t = 0
    TrajStates[i] = [mid_state]
    while True:
        ev = np.random.normal(K * Dtemp * Ctemp, SigmaEv)
        NexState, NextAction, R = TakeAction(st, 0, ev, Dtemp, Q)
        if NextAction == 0 or NextAction == 1:
            st = int(np.floor(len(States) / 2))
            RT[i] = t
            Choice[i] = Actions[NextAction] == Dtemp
            break

        else:
            st = NexState
            t = t + 1
            TrajStates[i].append(st)





RTMean = np.zeros(len(Coherence))
ACCMean = np.zeros(len(Coherence))
RTSEM = np.zeros(len(Coherence))
ACCSEM = np.zeros(len(Coherence))


for i in range(len(Coherence)):
    RTMean[i] = np.mean(RT[np.where(AllCoh == Coherence[i])])
    ACCMean[i] = np.mean(Choice[np.where(AllCoh == Coherence[i])])
    RTSEM[i] = np.std(RT[np.where(AllCoh == Coherence[i])])/np.sqrt(np.where(AllCoh == Coherence[i])[0].shape[0])
    ACCSEM[i] = np.std(Choice[np.where(AllCoh == Coherence[i])])/np.sqrt(np.where(AllCoh == Coherence[i])[0].shape[0])


Endval = np.zeros(len(AllCoh))
for i in range(len(TrajStates)):
    Endval[i] = TrajStates[i][-1]

## Inference from the closed form equations in sequential sampling
A = np.mean(np.abs(States[Endval.astype(int)]))
P_An = lambda C: 1 / (1 + np.exp(-2 * K * C * A))
RT_An = lambda C: (A / (K * C)) * np.tanh(K * C * A)



f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [1, 4]})
epsilon = sys.float_info.epsilon
xLim = 0.001
# plot the same data on both axes
zerosbin = np.linspace(-xLim/3, xLim/3, 100)
ax.plot(zerosbin, RT_An(epsilon)*np.ones(len(zerosbin)), 'b', linewidth=2, markersize=10)
X = np.linspace(Coherence[1] - 0.01, Coherence[5]+.1, 100)
ax2.plot(X, RT_An(X), c='b', lw=2)


ax.set_xlim(-xLim, xLim)
ax2.set_xlim(0.025, .6)
ax2.set_xscale('log')
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.get_yaxis().set_visible(False)
plt.setp(ax2, xticks=Coherence[1::2], xticklabels=['3.2', '12.8', '51.2'])
ax2.plot(Coherence, RTMean, 'k.', lw=2, markersize=10)
ax.plot(Coherence[0], RTMean[0], 'k.', lw=2, markersize=10)

ax2.errorbar(Coherence, RTMean, RTSEM, ls='none', marker='.', c='k')
ax.errorbar(Coherence[0], RTMean[0], RTSEM[0], ls='none', marker='.', c='k')

ax.plot(zerosbin, RT_An(epsilon)*np.ones(len(zerosbin)), 'b', linewidth=2, markersize=10)
plt.minorticks_off()
plt.setp(ax, xticks=[0], xticklabels=['0'])
# plt.xlabel('Coherence Level (%)')
ax.set_ylabel('RT (ms)')
f.text(0.5, 0.01, 'Coherence Level (%)', ha='center')


## Plot threshold during training
fig = plt.figure()
ax = plt.axes()

up_mean = np.empty((NumIter, NumTr))
up_mean[:] = np.nan

low_mean = np.empty((NumIter, NumTr))
low_mean[:] = np.nan

for iti in range(NumIter):
    BoundVals = States[AllBound[iti].astype(int)]
    UpBounds = BoundVals[BoundVals > 0]
    LowBound = BoundVals[BoundVals < 0]

    X = np.arange(0, BoundVals.shape[0])

    # ax.plot(X[BoundVals > 0], UpBounds, 'b.', alpha=0.15, ms=5)
    # ax.plot(X[BoundVals < 0], LowBound, 'r.', alpha=0.15, ms=5)

    tup = MySmooth(UpBounds, Bin)
    tlow = MySmooth(LowBound, Bin)

    ax.plot(X[BoundVals > 0], tup, 'b-', alpha=0.05, lw=2)
    ax.plot(X[BoundVals < 0], tlow, 'r-', alpha=0.05, lw=2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Training Progress (a.u.)')
    ax.set_ylabel('Terminal State (a.u.)')
    up_mean[iti, :len(tup)] = tup
    low_mean[iti, :len(tlow)] = tlow


ax.plot(X[::2], np.nanmean(up_mean, 0)[:int(NumTr/2)], c='b', lw=6, label='Right')
ax.plot(X[::2], np.nanmean(low_mean, 0)[:int(NumTr/2)], c='r', lw=6, label='Left')

ax.legend(loc='best', frameon=False, facecolor=None)


f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [1, 4]})
epsilon = sys.float_info.epsilon
xLim = 0.001
# plot the same data on both axes
zerosbin = np.linspace(-xLim/3, xLim/3, 100)
ax.plot(zerosbin, P_An(epsilon)*np.ones(len(zerosbin)), 'b', linewidth=2, markersize=10)
X = np.linspace(Coherence[1] - 0.01, Coherence[5]+.1, 100)
ax2.plot(X, P_An(X), c='b', lw=2)


ax.set_xlim(-xLim, xLim)
ax2.set_xlim(0.025, .6)
ax2.set_xscale('log')
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.get_yaxis().set_visible(False)
plt.setp(ax2, xticks=Coherence[1::2], xticklabels=['3.2', '12.8', '51.2'])
ax2.plot(Coherence, ACCMean, 'k.', lw=2, markersize=10)
ax.plot(Coherence[0], ACCMean[0], 'k.', lw=2, markersize=10)

ax2.errorbar(Coherence, ACCMean, ACCSEM, ls='none', marker='.', c='k')
ax.errorbar(Coherence[0], ACCMean[0], ACCSEM[0], ls='none', marker='.', c='k')

ax.plot(zerosbin, P_An(epsilon)*np.ones(len(zerosbin)), 'b', linewidth=2, markersize=10)
plt.minorticks_off()
plt.setp(ax, xticks=[0], xticklabels=['0'])
ax.set_ylabel('Accuracy (%)')
f.text(0.5, 0.01, 'Coherence Level (%)', ha='center')



## plot two state trajectory of two random trials
fig = plt.figure()
ax = plt.axes()
TmpCoh = AllCoh[0:100]*DiVector[0:100]
index1, index2 = np.where(100*TmpCoh == 6.4)[0][1], np.where(100*TmpCoh == -6.4)[0][3]
ax.plot(States[np.array(TrajStates[index1]).astype(int)], c='b', linewidth=2)
ax.plot(States[np.array(TrajStates[index2]).astype(int)], c='r', linewidth=2)
ax.hlines(0, 0, 1200, ls='--', colors='k')
plt.legend(['+6.4 %', '-6.4 %'], loc='best', frameon=False, facecolor=None)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('States')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


## Threshold distrbuation During Training
NumQ = 4
fig = plt.figure()
ax = fig.subplots(NumQ, 1)
ALLBB = np.zeros(NumTr)
ALLBB[::2] = np.nanmean(up_mean, 0)[:int(NumTr/2)]
ALLBB[1::2] = np.nanmean(low_mean, 0)[:int(NumTr/2)]

CutSize = int(len(ALLBB) / NumQ)
# NumBins = 20
Lims = 30
alphaweight = .1
yHight = 0.05
for t in range(NumQ):
    tBin = np.arange(t*CutSize, (t+1)*CutSize)
    AllBound = ALLBB[tBin]
    EnVal1 = np.zeros(len(AllBound))
    EnVal2 = np.zeros(len(AllBound))
    EnVal = np.zeros(len(AllBound))


    for i in range(len(AllBound)):
        if AllBound[i] > 0:
            EnVal1[i] = AllBound[i]
        else:
            EnVal2[i] = AllBound[i]
    EnVal1 = EnVal1[EnVal1 != 0]
    EnVal2 = EnVal2[EnVal2 != 0]


    sns.distplot(EnVal1, color='b', ax=ax[t], bins=5)

    sns.distplot(EnVal2, color='r', ax=ax[t], bins=5)



    ax[t].spines['right'].set_visible(False)
    ax[t].spines['top'].set_visible(False)
    if t != NumQ-1:
        ax[t].spines['bottom'].set_visible(False)
        ax[t].get_xaxis().set_visible(False)
    if t == 0:
        ax[t].legend(loc='best', frameon=False, facecolor=None)

plt.xlabel('States')



plt.show()
