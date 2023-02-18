import numpy as np
from riverswim import RiverSwim
from new_mdp_description import MDPDescription2
from tqdm import tqdm
from scipy.special import rel_entr
import matplotlib.pyplot as plt

T = 5000
ns = 20
na = 2
gamma = 0.99
UPDATE_FREQUENCY = 25
N_SIMS = 50

def dirichlet_sample(alphas):
    """
    Generate samples from an array of alpha distributions.
    """
    r = np.random.standard_gamma(alphas)
    return r / r.sum(-1, keepdims=True)


class PosteriorProbabilisties(object):
    def __init__(self, ns: int, na: int):
        self.ns = ns
        self.na = na

        self.prior_transition = np.ones((ns, na, ns))
        self.prior_rewards = 0.5*np.ones((ns, na, 2))
        self.n_visits_states = np.zeros((ns, na, ns))
        self.n_visits_rew = np.zeros((ns, na))


    def update(self, state: int, action: int, next_state: int, reward: float):
        self.n_visits_states[state, action, next_state] += 1
        self.n_visits_rew[state, action] += reward
        
    def sample_posterior(self):
        posterior_transition = self.prior_transition + self.n_visits_states
        posterior_rewards_alpha = self.prior_rewards[:, :, 0] + self.n_visits_rew
        posterior_rewards_beta = self.prior_rewards[:, :, 1] + self.n_visits_states.sum(-1) - self.n_visits_rew

        P = dirichlet_sample(posterior_transition)
        R = np.random.beta(posterior_rewards_alpha, posterior_rewards_beta)[..., np.newaxis]
        return P, R
    
    def mle(self):
        posterior_transition = self.prior_transition + self.n_visits_states
        posterior_rewards_alpha = self.prior_rewards[:, :, 0] + self.n_visits_rew
        posterior_rewards_beta = self.prior_rewards[:, :, 1] + self.n_visits_states.sum(-1) - self.n_visits_rew
        
        P = posterior_transition / posterior_transition.sum(-1, keepdims=True)
        R = posterior_rewards_alpha / (posterior_rewards_alpha + posterior_rewards_beta)
        return P, R[..., np.newaxis]


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


eval_no_sampling, posterior_no_sampling = run(T, False)
eval_sampling, posterior_sampling = run(T, True)

plt.plot(eval_no_sampling, label='no')
plt.plot(eval_sampling)
plt.legend()
plt.show()
import pdb
pdb.set_trace()