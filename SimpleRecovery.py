"""
Simple parameter-recovery pipeline for the RL-DDM.

Free / recovered parameters: K (drift scaling), lr (learning rate),
beta (softmax temperature), r_wait (cost of waiting per step).
Everything else is fixed.

For each of N_REP ground-truth parameter sets we:
  1) draw true (K, lr, beta, r_wait) from plausible ranges,
  2) simulate a stable "behavioural" target (averaged over several agents),
  3) refit the 4 params with gp_minimize, where the cost is averaged over
     several rollouts to denoise the (otherwise noisy) objective surface,
  4) store true vs. recovered.

Run:  python SimpleRecovery.py
Then: python PlotRecovery.py   (recovered-vs-true scatter + correlation matrix)
"""

import os
import numpy as np
import pickle
from scipy.optimize import dual_annealing

# Optimizer:
#   "cma"    - CMA-ES (gold standard for noisy behavioural fits; pip install cma)
#   "anneal" - scipy dual_annealing (no extra deps, noise-tolerant)
#   "gp"     - skopt Bayesian optimization (most sample-efficient; pip install skopt)
METHOD = "cma"

# ----------------------------- fixed model ---------------------------------
Gamma = 0.9          # discount factor
SigmaEv = 1.0        # evidence noise
States = np.linspace(-100, 100, 201)
Numstates = len(States)
mid = Numstates // 2                       # start state (zero evidence)
Actions = [-1, 1, 0]                       # left, right, wait
NumActions = len(Actions)
RewardCorrect, RewardWrong = 20.0, -50.0   # fixed payoffs

Coherence = np.array([0, 3.2, 6.4, 12.8, 25.6, 51.2]) / 100
NumTr = 300                                # trials per agent (50 per coherence)
MAXSTEP = 600                              # safeguard against runaway waiting

# ----------------------------- cost summary --------------------------------
NumQ = 3            # RT quantile bins
Numtime_Q = 3       # training-time bins (lets the cost see the learning curve)
RTMaxRange = 200.0

# ----------------------------- recovery setup ------------------------------
N_REP = 30                                 # ground-truth parameter sets
N_ROLL_FIT = 3                             # rollouts averaged per fit evaluation
N_ROLL_TARGET = 15                         # rollouts averaged for the target
N_CALLS = 120                              # objective evaluations per fit

# beta is fixed: behaviour saturates above ~10, so it is not identifiable here.
BETA_FIXED = 10.0

# (low, high) for both ground-truth draws and fit bounds (fitted params only)
BOUNDS = {
    "K":      (0.2, 0.8),
    "lr":     (0.02, 0.3),
    "r_wait": (-3.0, -0.3),
}
PARAM_NAMES = ["K", "lr", "r_wait"]
SKOPT_BOUNDS = [BOUNDS[p] for p in PARAM_NAMES]


def make_trials(seed):
    """Fixed trial sequence (directions + coherences) for one repetition."""
    rng = np.random.default_rng(seed)
    DiVector = np.hstack((np.ones(NumTr // 2), -1 * np.ones(NumTr // 2)))
    rng.shuffle(DiVector)
    AllCoh = np.repeat(Coherence, NumTr // len(Coherence))
    rng.shuffle(AllCoh)
    return AllCoh, DiVector


def simulate_agent(K, lr, beta, r_wait, AllCoh, DiVector, rng):
    """Run one learning agent through the trial sequence; return RT and ACC."""
    Q = np.zeros((Numstates, NumActions))
    Reward = [RewardCorrect, RewardWrong, r_wait]
    n = len(AllCoh)
    RT = np.zeros(n)
    ACC = np.zeros(n)

    for tr in range(n):
        D = DiVector[tr]
        C = AllCoh[tr]
        st = mid
        steps = 0
        while True:
            steps += 1
            ev = rng.normal(K * D * C, SigmaEv)
            qrow = Q[st]
            if np.ptp(qrow) == 0:                      # undifferentiated -> random
                a_idx = rng.integers(0, NumActions)
            else:
                p = np.exp(beta * (qrow - qrow.max()))  # max-subtraction = stable
                p = p / p.sum()
                a_idx = rng.choice(NumActions, p=p)
            action = Actions[a_idx]

            if action == 0:                            # wait: accumulate evidence
                nxt = int(np.clip(round(st + ev), 0, Numstates - 1))
                Q[st, a_idx] += lr * (Reward[2] + Gamma * Q[nxt].max() - Q[st, a_idx])
                st = nxt
                if steps >= MAXSTEP:                   # force a choice
                    action = 1 if States[st] >= 0 else -1
                    a_idx = Actions.index(action)
                    correct = action == D
                    Q[st, a_idx] += lr * (Reward[0 if correct else 1] - Q[st, a_idx])
                    RT[tr], ACC[tr] = steps, correct
                    break
            else:                                      # commit: terminal
                correct = action == D
                Q[st, a_idx] += lr * (Reward[0 if correct else 1] - Q[st, a_idx])
                RT[tr], ACC[tr] = steps, correct
                break
    return RT, ACC


def calc_probmat(RT, ACC):
    """Joint distribution over (training-time bin, RT quantile, correct/error)."""
    n = len(RT)
    P = np.zeros((Numtime_Q, NumQ, 2))
    tsize = n / Numtime_Q
    rsize = RTMaxRange / NumQ
    for ti in range(Numtime_Q):
        idx = np.arange(int(ti * tsize), int((ti + 1) * tsize))
        rt, acc = RT[idx], ACC[idx]
        for qi in range(NumQ):
            lo = qi * rsize
            hi = (qi + 1) * rsize if qi < NumQ - 1 else np.inf
            inbin = (rt >= lo) & (rt < hi)
            P[ti, qi, 0] = np.sum((acc == 1) & inbin) / len(rt)
            P[ti, qi, 1] = np.sum((acc == 0) & inbin) / len(rt)
    return P


def avg_probmat(K, lr, beta, r_wait, AllCoh, DiVector, rng, n_roll):
    mats = [calc_probmat(*simulate_agent(K, lr, beta, r_wait, AllCoh, DiVector, rng))
            for _ in range(n_roll)]
    return np.mean(mats, axis=0)


def run():
    ndim = len(SKOPT_BOUNDS)
    true_params = np.zeros((N_REP, ndim))
    fit_params = np.zeros((N_REP, ndim))

    for rep in range(N_REP):
        gt_rng = np.random.default_rng(1000 + rep)
        Kt = gt_rng.uniform(*BOUNDS["K"])
        lrt = gt_rng.uniform(*BOUNDS["lr"])
        rwt = gt_rng.uniform(*BOUNDS["r_wait"])
        true_params[rep] = [Kt, lrt, rwt]

        AllCoh, DiVector = make_trials(seed=2000 + rep)

        tgt_rng = np.random.default_rng(3000 + rep)
        Ptarget = avg_probmat(Kt, lrt, BETA_FIXED, rwt, AllCoh, DiVector,
                              tgt_rng, N_ROLL_TARGET)

        fit_rng = np.random.default_rng(4000 + rep)

        def objective(x):
            K, lr, r_wait = x
            Pmodel = avg_probmat(K, lr, BETA_FIXED, r_wait, AllCoh, DiVector,
                                 fit_rng, N_ROLL_FIT)
            return float(np.sum((Ptarget - Pmodel) ** 2))

        if METHOD == "cma":
            import cma
            lo = np.array([b[0] for b in SKOPT_BOUNDS])
            hi = np.array([b[1] for b in SKOPT_BOUNDS])
            # CMA-ES uses one step size for all dims -> optimise in [0,1] space
            es = cma.CMAEvolutionStrategy(
                [0.5] * ndim, 0.25,
                {"bounds": [0, 1], "maxfevals": N_CALLS, "seed": rep, "verbose": -9})
            while not es.stop():
                sols = es.ask()
                es.tell(sols, [objective(lo + s * (hi - lo)) for s in sols])
            best = lo + np.array(es.result.xbest) * (hi - lo)
        elif METHOD == "gp":
            from skopt import gp_minimize
            res = gp_minimize(objective, SKOPT_BOUNDS, n_calls=N_CALLS,
                              n_random_starts=10, random_state=rep)
            best = res.x
        else:  # "anneal": no local polishing -> robust to objective noise
            res = dual_annealing(objective, SKOPT_BOUNDS, maxfun=N_CALLS,
                                 no_local_search=True, seed=rep)
            best = res.x
        fit_params[rep] = best
        print(f"rep {rep:2d}  true={np.round(true_params[rep], 3)}  "
              f"fit={np.round(fit_params[rep], 3)}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SavedData")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Recovery_{METHOD}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump([true_params, fit_params, PARAM_NAMES], f)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    run()
