from riverswim import RiverSwim
import numpy as np
from scipy.special import rel_entr
from new_mdp_description import MDPDescription2
from tqdm import tqdm
from qucb import QUCB
from bpi import BPI
import matplotlib.pyplot as plt
import multiprocessing as mp
GAMMA = 0.99
NS = 30
NA = 2
C = 1e-3
P = 0.1
T = 50000
N_SIMS = 50

def run(T: int, type: str = 'QUCB'):
    env = RiverSwim(num_states=NS, min_reward=0.001)
    s = env.reset()
    
    true_mdp = MDPDescription2(env.transitions, env.rewards[..., np.newaxis], GAMMA, 1)
    #true_omega, _ = true_mdp.compute_allocation(navigation_constraints=True)
    V_greedy = true_mdp.V_greedy
    
    if type == 'QUCB':
        agent = QUCB(GAMMA, NS, NA, C, P)
    elif type == 'BPI':
        agent = BPI(GAMMA, NS, NA, C, P)

    compute_dist = lambda x, y: np.linalg.norm(x - y)
    eval = [compute_dist(agent.V, V_greedy)]


    for t in tqdm(range(T)):
        a = agent.forward(s)
        next_state, reward = env.step(a)
        agent.backward(s, a, reward, next_state)
        s = next_state
        
        eval.append(compute_dist(agent.V, V_greedy))

    return eval

def run_qucb(seed):
    np.random.seed(seed)
    return run(T, 'QUCB')
    
def run_bpi(seed):
    np.random.seed(seed)
    return run(T, 'BPI')

with mp.Pool(5) as p:
    eval_qucb = np.array(list(p.map(run_qucb, range(N_SIMS))))
    eval_bpi = np.array(list(p.map(run_bpi, range(N_SIMS))))

mu = eval_qucb.mean(0)
ce = 1.96*eval_qucb.std(0) / np.sqrt(N_SIMS)
plt.plot(range(T+1), mu, label='qucb')
plt.fill_between(range(T+1), mu -ce, mu+ce, alpha=0.3)

mu = eval_bpi.mean(0)
ce = 1.96*eval_bpi.std(0) / np.sqrt(N_SIMS)
plt.plot(range(T+1), mu, label='bpi')
plt.fill_between(range(T+1), mu -ce, mu+ce, alpha=0.3)
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()