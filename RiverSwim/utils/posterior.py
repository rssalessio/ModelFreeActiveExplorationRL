import numpy as np


def dirichlet_sample(alphas):
    """
    Generate samples from an array of alpha distributions.
    """
    r = np.random.standard_gamma(alphas)
    return r / r.sum(-1, keepdims=True)

class PosteriorProbabilisties(object):
    def __init__(self, ns: int, na: int, prior_p: float = .5, prior_r: float = .5):
        self.ns = ns
        self.na = na

        self.prior_transition = prior_p * np.ones((ns, na, ns))
        self.prior_rewards = prior_r * np.ones((ns, na, 2))
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