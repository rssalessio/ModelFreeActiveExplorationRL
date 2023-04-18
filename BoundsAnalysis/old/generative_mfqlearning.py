import numpy as np
from new_mdp_description import NewMDPDescription, BoundType
from tqdm import tqdm

def simulate_mfq_learning(mdp: NewMDPDescription, T: int, eta1: float, eta2: float, alpha: float):
    k = mdp.moment_order_k
    gamma = mdp.discount_factor
    ns, na = mdp.dim_state, mdp.dim_action
    
    Q = np.zeros((ns, na))
    M = np.zeros((ns, na))
    omega_o = np.ones((ns, na)) / (ns * na)
    num_visits = np.zeros((ns, na))
    
    for t in tqdm(range(T)):
        alpha_t = alpha ** t
        omega = alpha_t * np.ones((ns, na)) / (ns * na) + (1-alpha_t) * omega_o
        
        # Pick action
        action = num_visits.argmin() if num_visits.min() <= 2 else np.random.choice(ns*na, p=omega.flatten())
        state, action = (action // na, action % na)
        
        # Sample from MDP
        sampled_state = np.random.choice(ns, p=mdp.P[state, action])
        reward = mdp.R[state, action, sampled_state]
        
        # Update statistics
        num_visits[state, action] += 1
        delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
        deltap = ((delta / gamma) ** (2 * k)) - M[state, action]
        Q[state, action] += (1 / num_visits[state, action] ** eta1) * delta
        M[state, action] += (1 / num_visits[state, action] ** eta2) * deltap
        
        delta_sq = np.clip((Q.max(1)[:, np.newaxis] - Q) ** 2, a_min=1e-3, a_max=None)
        pi_greedy = Q.argmax(1)
        idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(na)] for s in range(ns)])
        delta_sq_subopt = delta_sq[idxs_subopt_actions]
        delta_sq_min =  delta_sq_subopt.min()
        omega_o, eval_res = mdp.compute_allocation(type= BoundType.BOUND_1, navigation_constraints=False,
                               Mk=M, Delta_sq=delta_sq, Delta_sq_min=delta_sq_min, pi_greedy=pi_greedy)
        #print(f'[{t}] {mdp.pi_greedy} - {pi_greedy}')
    print(f'[{t}] {mdp.pi_greedy} - {pi_greedy}')
    print(num_visits)
    print(omega_o)
    print(mdp.compute_allocation())
        

def simulate_q_learning(mdp: NewMDPDescription, T: int, eta1: float, eps: float):
    k = mdp.moment_order_k
    gamma = mdp.discount_factor
    ns, na = mdp.dim_state, mdp.dim_action
    
    Q = np.zeros((ns, na))
    num_visits = np.zeros((ns, na))
    
    for t in tqdm(range(T)):
        
        # Pick action
        action = num_visits.argmin()

        state, action = (action // na, action % na)
        
        # Sample from MDP
        sampled_state = np.random.choice(ns, p=mdp.P[state, action])
        reward = mdp.R[state, action, sampled_state]
        
        # Update statistics
        num_visits[state, action] += 1
        delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
        Q[state, action] += (1 / num_visits[state, action] ** eta1) * delta
    
        pi_greedy = Q.argmax(1)
    print(f'[{t}] {mdp.pi_greedy} - {pi_greedy}')
    print(num_visits)
    print(num_visits.sum())
    
    
if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
    ns, na = 5,3
    np.random.seed(2)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    k = 1
    gamma = 0.99
    alpha = 0.1
    eta1 = 0.6
    eta2 = 0.6
    eps = 0.9
    T = 1000
    mdp = NewMDPDescription(P, R, gamma, 1)
    simulate_q_learning(mdp, T, eta1, eps)
    simulate_mfq_learning(mdp, T, eta1, eta2, alpha)
        
    