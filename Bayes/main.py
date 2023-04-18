import numpy as np
from riverswim import RiverSwim
from new_mdp_description import MDPDescription2
from tqdm import tqdm
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle

T = 5000
ns = 20
na = 2
gamma = 0.99
UPDATE_FREQUENCY = 25
N_SIMS = 50
N_PROC = 5

def dirichlet_sample(alphas):
    """
    Generate samples from an array of alpha distributions.
    """
    r = np.random.standard_gamma(alphas)
    return r / r.sum(-1, keepdims=True)





def run(T: int, with_sampling: bool = False):
    env = RiverSwim(num_states=ns)
    s = env.reset()
    posterior = PosteriorProbabilisties(ns, na)
    
    true_mdp = MDPDescription2(env.transitions, env.rewards[..., np.newaxis], gamma, 1)
    true_omega, _ = true_mdp.compute_allocation(navigation_constraints=True)

    P, R = posterior.sample_posterior() if with_sampling else posterior.mle()
    mdp = MDPDescription2(P, R, gamma, 1)
    omega, _ = mdp.compute_allocation(navigation_constraints=True)
    
    compute_dist = lambda x, y: rel_entr(x, y).sum(-1).max()
    eval = [compute_dist(omega, true_omega)]


    for t in tqdm(range(T)):
        a = np.random.choice(2, p =omega[s] / omega[s].sum())
        next_state, reward = env.step(a)
        posterior.update(s, a, next_state, reward)
        s = next_state
        
        if t % UPDATE_FREQUENCY == 0:
            P, R = posterior.sample_posterior() if with_sampling else posterior.mle()
            mdp = MDPDescription2(P, R, gamma, 1)
            omega, _ = mdp.compute_allocation(navigation_constraints=True)
            eval.append(compute_dist(omega, true_omega))

    return eval, posterior

def run_no_sampling(seed: int):
    print(f'Id: {seed}')
    np.random.seed(seed)
    return run(T, False)

def run_sampling(seed: int):
    print(f'Id: {seed}')
    np.random.seed(seed)
    return run(T, True)


with mp.Pool(N_PROC) as p:
    print('No sampling')
    res_no_sampling = p.map(run_no_sampling, range(N_SIMS))
    print('Sampling')
    res_sampling = p.map(run_sampling, range(N_SIMS))

with open('data.pkl', 'wb') as f:
    pickle.dump(
        {
            'ns': ns, 'na': na, 'T': T, 'gamma': gamma, 'update_frequency': UPDATE_FREQUENCY,
            'res_no_sampling': res_no_sampling, 'res_sampling': res_sampling
        }, f
    )