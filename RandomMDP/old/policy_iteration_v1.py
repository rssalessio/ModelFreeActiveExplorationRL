import numpy as np
import numpy.typing as npt
import cvxpy as cp
from new_mdp_description import NewMDPDescription
from utils import policy_iteration, policy_evaluation, compute_state_action_distribution_from_policy
from scipy.optimize import minimize
from jax import grad, jit, hessian
import jax.numpy as jnp
golden_ratio_sq =  ((1+ np.sqrt(5)) / 2) ** 2


def explorative_policy_evaluation(
    gammae: float,
    pi: npt.NDArray[np.float64],
    mdp: NewMDPDescription,
    abs_tol: float = 1e-6):
    ns, na = mdp.dim_state, mdp.dim_action
    
    
    H0 = (2 + 8 * golden_ratio_sq * mdp.Mk_V_greedy[mdp.idxs_subopt_actions].reshape(ns, na-1)) / mdp.delta_sq[mdp.idxs_subopt_actions].reshape(ns, na-1)
    H1 = (2 + 8 * golden_ratio_sq * mdp.Mk_V_greedy[~mdp.idxs_subopt_actions]) / (mdp.delta_sq_min * (1-mdp.discount_factor)**2)

    H0 = H0 * mdp.normalizer
    H1 = H1 * mdp.normalizer
    
    
    eta = (H1 / pi[~mdp.idxs_subopt_actions]).flatten() #+ np.max(H0 / pi[mdp.idxs_subopt_actions].reshape(ns, na-1), 1)
    Ve = np.zeros(ns)
    
    while True:
        Delta = 0
        V_next = np.array([ 1/eta[s] + gammae * pi[s] @ (mdp.P[s] @ Ve) for s in range(ns)])
        Delta = np.max([Delta, np.abs(V_next - Ve).max()])
        Ve = V_next
        if Delta < abs_tol:
            break
    return Ve / ( mdp.normalizer)


def explorative_policy_improvement(
    gammae: float,
    pi: npt.NDArray[np.float64],
    mdp: NewMDPDescription,
    abs_tol: float = 1e-6):
    
    ns, na = mdp.dim_state, mdp.dim_action
    H0 = (2 + 8 * golden_ratio_sq * mdp.Mk_V_greedy[mdp.idxs_subopt_actions].reshape(ns, na-1)) / mdp.delta_sq[mdp.idxs_subopt_actions].reshape(ns, na-1)

    H1 = (2 + 8 * golden_ratio_sq * mdp.Mk_V_greedy[~mdp.idxs_subopt_actions])[:, np.newaxis] / (mdp.delta_sq[mdp.idxs_subopt_actions].reshape(ns, na-1) * (1-mdp.discount_factor)**2)

    H0 = H0 * mdp.normalizer
    H1 = H1 * mdp.normalizer
    
    Ve = explorative_policy_evaluation(gammae, pi, mdp) * mdp.normalizer
    G_sa = (mdp.P @ Ve)
    pi_new = np.zeros((ns, na))
    
    
    
    
    for s in range(ns):
        # import pdb
        # pdb.set_trace()
        def objective(x):
            x = jnp.array(x)
            U = jnp.max(H0[s]/x[mdp.idxs_subopt_actions[s]]+H1[s]/ (x[~mdp.idxs_subopt_actions[s]]))

            return (1/U) + gammae * (jnp.sum(jnp.multiply(x, G_sa[s])))
        
        obj_grad = jit(grad(objective), backend='cpu')
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x)-1, 'jac': lambda x: np.ones(na)},
        ]
        
        res = minimize(
            fun = lambda x: np.asarray(objective(x)).item(),
            x0 = np.ones(na)/na,
            method = 'SLSQP',
            jac = lambda x: np.asarray(obj_grad(x)),
            bounds=[(1e-2,1) for _ in range(na)],
            constraints=constraints,
            options={'disp': False}
        )
        
        # omega = cp.Variable(na, nonneg=True)
        
        # T1 = H1[s]*cp.inv_pos(omega[~mdp.idxs_subopt_actions[s]])
 
        # T2 = cp.max(cp.multiply(cp.inv_pos(omega[mdp.idxs_subopt_actions[s]]), H0[s]))
        
        # obj = (1-gammae)*(T1+T2) +  gammae * (cp.sum(cp.multiply(omega, G_sa[s])))
        # constraints = [cp.sum(omega) == 1]
 
        # problem = cp.Problem(cp.Minimize(obj), constraints)
        # result = problem.solve()
        #print(f'{s}: {res.fun} - {res.x}')
        pi_new[s] = res.x
        #Ve = explorative_policy_evaluation(gammae, pi_new, mdp) * mdp.normalizer
        #G_sa = (mdp.P @ Ve)
    
    print(pi_new)
    import pdb
    pdb.set_trace()
    return pi_new
    

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ns, na = 5,3
    np.random.seed(2)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    
    discount_factor = 0.99
    gammae = 0.01
    
    mdp = NewMDPDescription(P, R, discount_factor, 1)
    pi, val_pi = mdp.compute_allocation(navigation_constraints=True)
    print(pi)
    print(val_pi)
    print(pi / pi.sum(-1)[:, np.newaxis])

    pinew = np.ones((ns,na))/na
    for i in range(5):
        val_pinew = explorative_policy_evaluation(gammae, pinew, mdp)
        pinew = explorative_policy_improvement(gammae, pinew, mdp)
    print(f'{val_pinew.max()} - {pinew}')
    
    omega_new = compute_state_action_distribution_from_policy(pinew, mdp.P)
    print(omega_new)
    print(val_pi)
    print(mdp.evaluate_allocation(omega_new, navigation_constraints=True))