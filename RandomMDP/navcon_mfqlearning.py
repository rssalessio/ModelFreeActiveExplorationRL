import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
from new_mdp_description import MDPDescription2, BoundType, MDPDescription
from tqdm import tqdm
from collections import deque
from utils import project_omega, compute_stationary_distribution, policy_evaluation
golden_ratio_sq =  ((1+ np.sqrt(5)) / 2) ** 2


def simulate_mfq_learning(mdp: MDPDescription2, T: int, eta1: float, eta2: float, eps: float, frequency_eval: int = 25, nav_constr: bool = False):
    k = mdp.moment_order_k
    gamma = mdp.discount_factor
    ns, na = mdp.dim_state, mdp.dim_action
    
    Q = np.zeros((ns, na)) #* (1/(1-mdp.discount_factor))
    M = np.zeros((ns, na)) #* (1/(1-mdp.discount_factor)**2)
    omega_o = np.ones((ns, na)) / (ns * na)
    #policy = np.ones((ns, na)) / na
    num_visits = np.zeros((ns, na))
    last_visit = np.zeros((ns, na))
    
    state = np.random.choice(ns)    
    #omega_nav_constr = mdp.compute_allocation(navigation_constraints=True)
    values = []
    
    
    for t in tqdm(range(T)):
        omega = eps * np.ones((na)) / (na) + (1-eps) * (omega_o[state]/omega_o[state].sum())
        # Pick action
        action = num_visits[state].argmin() if num_visits[state].min() <= 2 else np.random.choice(na, p=omega.flatten())
        
        
        last_visit[state, action] = t
        # Sample from MDP
        sampled_state = np.random.choice(ns, p=mdp.P[state, action])
        reward = mdp.R[state, action, sampled_state]
        
        # Update statistics
        num_visits[state, action] += 1
        delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
        Q[state, action] += (1 / num_visits[state, action] ** eta1) * delta
        
        delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
        deltap = ((delta / gamma) ** (2 * k)) - M[state, action]
        M[state, action] += (1 / num_visits[state, action] ** eta2) * deltap
        
        if t % frequency_eval == 0:
            delta_sq = np.clip((Q.max(1)[:, np.newaxis] - Q) ** 2, a_min=1e-9, a_max=None)
            pi_greedy = Q.argmax(1)
            idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(na)] for s in range(ns)])
            delta_sq_subopt = delta_sq[idxs_subopt_actions]
            delta_sq_min =  delta_sq_subopt.min()
            omega_o, eval_res = mdp.compute_allocation(type= BoundType.BOUND_1, navigation_constraints=nav_constr,
                                Mk=M, Delta_sq=delta_sq, Delta_sq_min=delta_sq_min, pi_greedy=pi_greedy, num_visits=None)

            value = policy_evaluation(mdp.discount_factor, mdp.P, mdp.R, pi_greedy, atol=1e-4)
            value_err = np.linalg.norm(value - mdp.V_greedy, 2)
            values.append((t, value_err))

        state = sampled_state
        
    return Q, M, values, num_visits, last_visit
    

def simulate_soft_learning(mdp: MDPDescription2, T: int, eta1: float, eta2: float, eta3: float, theta: float, frequency_eval: int = 25, nav_constr: bool = False):
    k = mdp.moment_order_k
    gamma = mdp.discount_factor
    ns, na = mdp.dim_state, mdp.dim_action
    
    Q = np.zeros((ns, na)) #*# (1/(1-mdp.discount_factor))
    M = np.zeros((ns, na)) #* (1/(1-mdp.discount_factor)**2)
    Qexploration = np.zeros((ns, na))
    num_visits = np.zeros((ns, na))
    last_visit = np.zeros((ns, na))
    
    state = np.random.choice(ns)
    values = []
    
    H = np.log(1/(1-mdp.discount_factor)) / np.log(1/mdp.discount_factor)
    
    def get_policy(s):
        #V = theta * np.log(np.exp((Qexploration[s] -Qexploration[s].max())/theta).sum(-1))
        policy = np.exp((Qexploration[s] - Qexploration[s].max())/theta)
        policy = policy/policy.sum()
        return policy
    
    action = np.random.choice(na, p=get_policy(state).flatten())
    
    for t in tqdm(range(T)):
        #action = np.random.choice(na, p=get_policy(state).flatten())
        
        # Sample from MDP
        sampled_state = np.random.choice(ns, p=mdp.P[state, action])
        reward = mdp.R[state, action, sampled_state]
        
        # Update statistics
        num_visits[state, action] += 1
        last_visit[state, action] = t
        delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
        Q[state, action] += (1 / num_visits[state, action] ** eta1) * delta
        
        delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
        deltap = ((delta / gamma) ** (2 * k)) - M[state, action]
        M[state, action] += (1 / num_visits[state, action] ** eta2) * deltap
        
        
        delta_sq = np.clip((Q.max(1)[:, np.newaxis] - Q) ** 2, a_min=1e-9, a_max=None)
        pi_greedy = Q.argmax(1)
        idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(na)] for s in range(ns)])
        delta_sq_subopt = delta_sq[idxs_subopt_actions]
        r_qexp = -np.maximum(delta_sq, delta_sq_subopt.min()) / (2 + 8 *golden_ratio_sq * M[state])
        
        if action == Q[state].argmax():
            r_qexp *= (1-mdp.discount_factor)**2
        
        #policy = get_policy(sampled_state)
        #next_V = theta * np.log(np.exp((Qexploration[sampled_state]-Qexploration[sampled_state].max()) / theta).sum()) # policy * (-theta*np.log(policy) + Qexploration[sampled_state])
        
        
        policy = get_policy(sampled_state).flatten()
        next_action = np.random.choice(na, p=policy)
        inner = np.exp((Qexploration[sampled_state]-Qexploration[sampled_state].max()) / theta) #/ policy
        
        next_V = theta * np.log((inner[next_action] / policy[next_action])) #+ Qexploration[sampled_state].max()
        
        delta_qexp = r_qexp[state,action]  + mdp.discount_factor * next_V - Qexploration[state, action]
        Qexploration[state, action] +=  (1 / num_visits[state, action] ** eta3) * delta_qexp
        
        
        if t % frequency_eval == 0:
            pi_greedy = Q.argmax(1)
         
            value = policy_evaluation(mdp.discount_factor, mdp.P, mdp.R, pi_greedy, atol=1e-4)
            value_err = np.linalg.norm(value - mdp.V_greedy, 2)
            values.append((t, value_err))

        state = sampled_state
        action = next_action
        
    return Q, M, values, num_visits, last_visit
    # print(f'[{t}] {mdp.pi_greedy} - {pi_greedy}')
    # #print(num_visits)
    # print(omega_o)
    # print(omega)
    
    # # mdp = MDPDescription(mdp.P, mdp.R, mdp.discount_factor)
    
    
    # print(num_visits/num_visits.sum())
    # print(omega_nav_constr[0])
    
    # omega = alpha_t * np.ones((ns,na)) / (ns*na) + (1-alpha_t) * omega_o
    # omegao_gen_proj = project_omega(omega_o, mdp.P, tol=1e-12)
    # omega_gen_proj = project_omega(omega, mdp.P, tol=1e-12)
    
    # omega2 = alpha_t * np.ones((ns,na)) / (na) + (1-alpha_t) * policy
    # omega2 = compute_stationary_distribution(omega2, mdp.P)
    # # piW = np.eye(na)[W.argmax(1)] # convert one hot encoding
    # # omega_new = 0.3*piW/piW.sum() + 0.7 * omega
    # # omega_new_proj = project_omega(omega_new, mdp.ansa.it/?refresh_ceP, tol=1e-4)
    # import pdb
    
    # pdb.set_trace()
    
        

def simulate_q_learning(mdp: MDPDescription2, T: int, eta: float, eps: float, frequency_eval: int = 25):
    gamma = mdp.discount_factor
    ns, na = mdp.dim_state, mdp.dim_action
    
    Q = np.zeros((ns, na)) #* (1/(1-mdp.discount_factor))
    last_visit = np.zeros((ns, na))
    num_visits = np.zeros((ns, na))
    state = np.random.choice(ns)
    values = []
    
    for t in tqdm(range(T)):
        # Pick action
        action = np.random.choice(na) if np.random.uniform() < eps else Q[state].argmax()
        
        # Sample from MDP
        sampled_state = np.random.choice(ns, p=mdp.P[state, action])
        reward = mdp.R[state, action, sampled_state]
        last_visit[state, action] = t
        # Update statistics
        num_visits[state, action] += 1
        delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
        Q[state, action] += (1 / num_visits[state, action] ** eta) * delta
    
        state = sampled_state
        if t % frequency_eval == 0:
            pi_greedy = Q.argmax(1)
            value = policy_evaluation(mdp.discount_factor, mdp.P, mdp.R, pi_greedy, atol=1e-4)
            value_err = np.linalg.norm(value - mdp.V_greedy, 2)
            values.append((t, value_err))
    
    return Q, values, num_visits, last_visit
    
    
if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
    N_SIMS = 20
    NS = 10
    
    FREQUENCY_EVAL = 100
    
    for NA in [15]:
        np.random.seed(2)
        P = np.random.dirichlet(np.ones(NS), size=(NS, NA))
        R = np.random.dirichlet(np.ones(NS), size=(NS, NA))
        k = 1
        gamma = 0.99
        eta1 = 0.5
        eta2 = 0.6
        eta3 = 0.7
        eps = 0.3
        theta = 1
        T = 20000
        mdp = MDPDescription2(P, R, gamma, 1)
        
        def _sim_qlearning(seed):
            np.random.seed(seed)
            return simulate_q_learning(mdp, T, eta1, eps, FREQUENCY_EVAL)
        def _sim_uniform(seed):
            np.random.seed(seed)
            return simulate_q_learning(mdp, T, eta1, 1, FREQUENCY_EVAL)
        
        def _sim_mflearning(seed):
            np.random.seed(seed)
            return simulate_mfq_learning(mdp, T, eta1, eta2, eps, FREQUENCY_EVAL)
        
        def _sim_mfclearning(seed):
            np.random.seed(seed)
            return simulate_mfq_learning(mdp, T, eta1, eta2, eps, FREQUENCY_EVAL, True)
        
        
        def _sim_softlearning(seed):
            np.random.seed(seed)
            return simulate_soft_learning(mdp, T, eta1, eta2, eta3, theta, FREQUENCY_EVAL, True)
        
        # def _sim_softlearning2(seed):
        #     np.random.seed(seed)
        #     return simulate_soft_learning(mdp, T, eta1, eta2, eta3, 1, FREQUENCY_EVAL, True)
        
        # def _sim_softlearning3(seed):
        #     np.random.seed(seed)
        #     return simulate_soft_learning(mdp, T, eta1, eta2, eta3, 100, FREQUENCY_EVAL, True)
        
        
        with mp.Pool(5) as p:
            res_q_learning = p.map(_sim_qlearning, range(N_SIMS))
            res_uniform_learning = p.map(_sim_uniform, range(N_SIMS))
            res_soft_learning = p.map(_sim_softlearning, range(N_SIMS))
            # res_soft_learning2 = p.map(_sim_softlearning2, range(N_SIMS))
            # res_soft_learning3 = p.map(_sim_softlearning3, range(N_SIMS))
            #res_mf_learning = p.map(_sim_mflearning, range(N_SIMS))
            #res_mfc_learning = p.map(_sim_mfclearning, range(N_SIMS))
        
        
        with open(f'navcon_q_comparison_S{NS}_A{NA}_final2.pkl', 'wb') as f:
            pickle.dump({
                'res_uniform_learning': res_uniform_learning,
                'res_q_learning': res_q_learning,
                'res_soft_learning': res_soft_learning,
                # 'res_soft_learning2': res_soft_learning2,
                # 'res_soft_learning3': res_soft_learning3,
                #'res_mf_learning': res_mf_learning,
                #'res_mfc_learning': res_mfc_learning
                }, f,
                protocol=pickle.HIGHEST_PROTOCOL)

    # res_q_learning = np.array([x[1] for x in res_q_learning])
    # res_mf_learning = np.array([x[2] for x in res_mf_learning])
    
    # plt.plot(res_q_learning.mean(0)[:,0], res_q_learning.mean(0)[:, 1], label='q')
    # plt.plot(res_mf_learning.mean(0)[:,0], res_mf_learning.mean(0)[:, 1])
    # #plt.xscale('log')
    # plt.legend()
    # plt.show()
    # import pdb
    # pdb.set_trace()
    
    #simulate_mfq_learning(mdp, T, eta1, eta2, alpha)
        

# def simulate_mfq_learning(mdp: MDPDescription2, T: int, eta1: float, eta2: float, alpha: float):
#     k = mdp.moment_order_k
#     gamma = mdp.discount_factor
#     ns, na = mdp.dim_state, mdp.dim_action
    
#     Q = np.zeros((ns, na))
#     M = np.zeros((ns, na))
#     W = np.zeros((ns, na))
#     omega_o = np.ones((ns, na)) / (ns * na)
#     policy = np.ones((ns, na)) / na
#     num_visits = np.zeros((ns, na))
    
#     state = np.random.choice(ns)
#     buffer = deque(maxlen=50)
#     buffer.append(state)
    
#     omega_nav_constr = mdp.compute_allocation(navigation_constraints=True)
#     for t in tqdm(range(T)):
        
#         alpha_t = alpha # max(0.1, alpha ** t)
        
#         omega = alpha_t * np.ones((na)) / (na) + (1-alpha_t) * (omega_o[state]/omega_o[state].sum()) #policy
#         #piW = np.eye(na)[W.argmax(1)] # convert one hot encoding
#         #omega = 0.3*(piW[state]/piW[state].sum()) + 0.7 * omega
#         # Pick action
#         action = num_visits[state].argmin() if num_visits[state].min() <= 2 else np.random.choice(na, p=omega.flatten())
        
#         # Sample from MDP
#         sampled_state = np.random.choice(ns, p=mdp.P[state, action])
#         reward = mdp.R[state, action, sampled_state]
        
#         # Update statistics
#         num_visits[state, action] += 1
#         delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
#         Q[state, action] += (1 / num_visits[state, action] ** eta1) * delta
        
#         delta = reward + gamma * Q[sampled_state].max() - Q[state, action]
#         deltap = ((delta / gamma) ** (2 * k)) - M[state, action]
#         M[state, action] += (1 / num_visits[state, action] ** eta2) * deltap
        
#         # buffer.append(sampled_state)
#         # if len(buffer) > 30:
#         Qucb = Q# + np.sqrt(np.log(num_visits.sum()) / (1+num_visits))
#         delta_sq = np.clip((Qucb.max(1)[:, np.newaxis] - Qucb) ** 2, a_min=1e-3, a_max=None)
#         pi_greedy = Qucb.argmax(1)
#         idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(na)] for s in range(ns)])
#         delta_sq_subopt = delta_sq[idxs_subopt_actions]
#         delta_sq_min =  delta_sq_subopt.min()
#         omega_o, eval_res = mdp.compute_allocation(type= BoundType.BOUND_1, navigation_constraints=False,
#                                Mk=M, Delta_sq=delta_sq, Delta_sq_min=delta_sq_min, pi_greedy=pi_greedy, num_visits=None)
#         # policy = mdp.compute_policy_v1(
#         #     Mk=M, Delta_sq=delta_sq, Delta_sq_min=delta_sq_min, pi_greedy=pi_greedy
#         # )[0]
#         # omega_o, eval_res = mdp.compute_allocation(type= BoundType.BOUND_1, navigation_constraints=True,
#         #                        Mk=M, Delta_sq=delta_sq, Delta_sq_min=delta_sq_min, pi_greedy=pi_greedy)
#         #print(f'[{t}] {mdp.pi_greedy} - {pi_greedy}')
        
#         # omega = alpha_t * np.ones((na)) / (na) + (1-alpha_t) * (omega_o[sampled_state]/omega_o[sampled_state].sum())
#         # piW = np.eye(na)[W.argmax(1)] # convert one hot encoding
#         # omega = 0.3*(piW[sampled_state]/piW[sampled_state].sum()) + 0.7 * omega
#         # # Pick action
#         # action_new = num_visits[sampled_state].argmin() if num_visits[sampled_state].min() <= 2 else np.random.choice(na, p=omega.flatten())
        
#         # delta = np.nan_to_num(-delta_sq[state, action]/(1e-3+M[state, action])) + gamma * W[sampled_state, action_new] - W[state, action]
#         # W[state, action] += (1 / num_visits[state, action] ** eta1) * delta
#         state = sampled_state
#         # action = action_new
        
        
        
    
    
#     print(f'[{t}] {mdp.pi_greedy} - {pi_greedy}')
#     #print(num_visits)
#     print(omega_o)
#     print(omega)
    
#     # mdp = MDPDescription(mdp.P, mdp.R, mdp.discount_factor)
    
    
#     print(num_visits/num_visits.sum())
#     print(omega_nav_constr[0])
    
#     omega = alpha_t * np.ones((ns,na)) / (ns*na) + (1-alpha_t) * omega_o
#     omegao_gen_proj = project_omega(omega_o, mdp.P, tol=1e-12)
#     omega_gen_proj = project_omega(omega, mdp.P, tol=1e-12)
    
#     omega2 = alpha_t * np.ones((ns,na)) / (na) + (1-alpha_t) * policy
#     omega2 = compute_stationary_distribution(omega2, mdp.P)
#     # piW = np.eye(na)[W.argmax(1)] # convert one hot encoding
#     # omega_new = 0.3*piW/piW.sum() + 0.7 * omega
#     # omega_new_proj = project_omega(omega_new, mdp.ansa.it/?refresh_ceP, tol=1e-4)
#     import pdb
    
#     pdb.set_trace()
    
        