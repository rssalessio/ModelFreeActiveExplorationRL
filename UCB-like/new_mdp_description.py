import numpy as np
import numpy.typing as npt
import cvxpy as cp
from utils import policy_iteration, compute_stationary_distribution, soft_policy_iteration
from typing import Tuple, Optional
from mdp_description import MDPDescription
from enum import Enum

class BoundType(Enum):
    BOUND_1 = 1,
    BOUND_2 = 2


golden_ratio =  ((1+ np.sqrt(5)) / 2) ** 2

class MDPDescription2(MDPDescription):
    V_greedy_k: npt.NDArray[np.float64]
    Mk_V_greedy: npt.NDArray[np.float64]
    moment_order_k: int
    
    def __init__(self, P: npt.NDArray[np.float64], R: npt.NDArray[np.float64], discount_factor: float, moment_order_k: int, abs_tol: float = 1e-6):
        super().__init__(P, R, discount_factor, abs_tol)
        self.moment_order_k = moment_order_k
        
        self.V_greedy_k = (self.V_greedy[:, np.newaxis, np.newaxis] - self.avg_V_greedy[np.newaxis,:,:]) ** (2 * self.moment_order_k)
        self.Mk_V_greedy = (
            P.reshape(self.dim_state * self.dim_action, -1) 
            * (self.V_greedy_k.reshape(self.dim_state, self.dim_state*self.dim_action).T)
        ).sum(-1).reshape(self.dim_state, self.dim_action) ** (2 ** (1 - self.moment_order_k))
    
    def evaluate_allocation(self, omega: npt.NDArray[np.float64], type: BoundType = BoundType.BOUND_1, navigation_constraints: bool = False) -> float:
        if navigation_constraints is True:
            checks = np.array([np.sum(omega[s]) - np.sum(np.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)])
            assert np.all(np.isclose(checks, 0)), "Allocation does not satisfy navigation constraints"
        ns, na = self.dim_state, self.dim_action
        
        if type == BoundType.BOUND_1:
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * (2 + 8 * golden_ratio * self.Mk_V_greedy[s,a]) / (omega[s,a] * self.delta_sq[s,a])
                    
                    obj_supp_s = []
                    for sp in range(ns):
                        T2 = self.normalizer * (2 + 8 * golden_ratio * self.Mk_V_greedy[sp, self.pi_greedy[sp]]) / (omega[sp, self.pi_greedy[sp]] * self.delta_sq[s,a] * ((1 - self.discount_factor) ** 2))
                        obj_supp_s.append(T2)
                    
                    obj_supp.append(T1 + np.max(obj_supp_s))

            objective = np.max(obj_supp)
            return objective / self.normalizer
        
        elif type == BoundType.BOUND_2:
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * (2 + 8 * golden_ratio * self.Mk_V_greedy[s,a]) / (omega[s,a] * self.delta_sq[s,a])
                
                    obj_supp.append(T1)
        
            obj_supp_s = []
            for sp in range(ns):
                T2 = self.normalizer * (2 + 8 * golden_ratio * self.Mk_V_greedy[sp, self.pi_greedy[sp]]) / (omega[sp, self.pi_greedy[sp]] * self.delta_sq_min * ((1 - self.discount_factor) ** 2))
                obj_supp_s.append(T2)
            
                        
            objective = np.max(obj_supp) + np.max(obj_supp_s)
            return objective / self.normalizer
        
            # for s in range(ns):
            #     obj_supp = []
                
            #     T2 = self.normalizer *  (2 + 8 * golden_ratio * self.Mk_V_greedy[s, self.pi_greedy[s]]) / (omega[s, self.pi_greedy[s]] * self.delta_sq_min * ((1 - self.discount_factor) ** 2))

            #     for a in range(na):
            #         if a == self.pi_greedy[s]: continue
            #         T1 = self.normalizer * (2 + 8 * golden_ratio * self.Mk_V_greedy[s,a]) / (omega[s,a] * (self.delta_sq[s,a]))
            #         obj_supp.append(T1)
                    
            #     objective += T2 + np.max(obj_supp)

            # return objective / self.normalizer
        else:
            raise Exception(f'Type {type} not found')
        
    
    def compute_allocation(
            self,
            type: BoundType = BoundType.BOUND_1,
            navigation_constraints: bool = False,
            Mk: Optional[npt.NDArray[np.float64]] = None,
            Delta_sq: Optional[npt.NDArray[np.float64]] = None,
            Delta_sq_min: Optional[float] = None,
            pi_greedy: Optional[npt.NDArray[np.float64]] = None,
            num_visits: Optional[npt.NDArray[np.float64]] = None,     
            ) -> Tuple[npt.NDArray[np.float64], float]:
    
        if Mk is None:
            Mk = self.Mk_V_greedy
        if Delta_sq is None:
            Delta_sq = self.delta_sq
        if Delta_sq_min is None:
            Delta_sq_min = self.delta_sq_min
        if pi_greedy is None:
            pi_greedy = self.pi_greedy
        
        if num_visits is not None:
            Mk += (1-self.discount_factor) / np.sqrt(num_visits + 1)
            Delta_sq_min = max(1e-7, Delta_sq_min - 1 / np.sqrt(num_visits + 1).sum())
            Delta_sq = np.maximum(Delta_sq_min, Delta_sq - 1 / np.sqrt(num_visits + 1))
            
            
        
        ns, na = self.dim_state, self.dim_action
        omega = cp.Variable((ns, na), nonneg=True)
        sigma = cp.Variable(1, nonneg=True)
        constraints = [cp.sum(omega) == 1, omega >= sigma, sigma >= 1e-15]
        
        if navigation_constraints:
            constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)])

        tol = 0#1e-14
        def upper_bound_eq_5():
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * cp.inv_pos(omega[s,a]) * (2 + 8 * golden_ratio * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    
                    obj_supp_s = []
                    for sp in range(ns):
                        T2 = self.normalizer * cp.inv_pos(omega[sp, pi_greedy[sp]]) * (2 + 8 * golden_ratio * Mk[sp, pi_greedy[sp]]) / (tol + Delta_sq[s,a] * ((1 - self.discount_factor) ** 2))

                        obj_supp_s.append(T2)
                    
                    obj_supp.append(T1 + cp.max(cp.hstack(obj_supp_s)))

            objective = cp.max(cp.hstack(obj_supp))
            return objective

        def upper_bound_eq_6():
            objective = 0
            for s in range(ns):
                obj_supp = []
                
                T2 = self.normalizer * cp.inv_pos(omega[s, pi_greedy[s]]) * (2 + 8 * golden_ratio * Mk[s, pi_greedy[s]]) /  (tol+Delta_sq_min * ((1 - self.discount_factor) ** 2))
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * cp.inv_pos(omega[s,a] ) * (2 + 8 * golden_ratio * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    obj_supp.append(T1)
                    
                objective += T2 + cp.max(cp.hstack(obj_supp))

            return objective
        solver = cp.MOSEK
        for _it in range(10):
            try:
                if type == BoundType.BOUND_1:
                    objective = cp.Minimize(upper_bound_eq_5())
                elif type == BoundType.BOUND_2:
                    objective = cp.Minimize(upper_bound_eq_6())
                else:
                    raise Exception(f'Type {type} not found')

                problem = cp.Problem(objective, constraints)
                result = problem.solve(verbose=False, solver=solver, warm_start=True)#, abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=200)
                break
            except Exception as e:
                solver = np.random.choice([cp.MOSEK, cp.ECOS, cp.SCS])
                if _it == 9:
                    raise Exception(f'Cannot solve the UB! {e}')
        return omega.value, self.evaluate_allocation(omega.value, type, navigation_constraints=False)


    def compute_policy_v1(
            self,
            Mk: Optional[npt.NDArray[np.float64]] = None,
            Delta_sq: Optional[npt.NDArray[np.float64]] = None,
            Delta_sq_min: Optional[float] = None,
            pi_greedy: Optional[npt.NDArray[np.float64]] = None        
            ) -> Tuple[npt.NDArray[np.float64], float]:
    
        if Mk is None:
            Mk = self.Mk_V_greedy
        if Delta_sq is None:
            Delta_sq = self.delta_sq
        if Delta_sq_min is None:
            Delta_sq_min = self.delta_sq_min
        if pi_greedy is None:
            pi_greedy = self.pi_greedy
        ns, na = self.dim_state, self.dim_action
        policy = np.zeros((ns, na))
        
        tol = 0#1e-14
        
        
        for s in range(ns):
        #     for a in range(na):
        #         if a == pi_greedy[s]: continue
        #         else:
        #             policy[s,a] = (2 + 8 * golden_ratio * Mk[s,a]) / (Delta_sq[s,a]+tol)
        #     T = (2 + 8 * golden_ratio * Mk[s,pi_greedy[s]]) / (Delta_sq_min+tol)
        #     policy[s, pi_greedy[s]] = np.sqrt(T * policy[s].sum())
            
        # policy = policy / policy.sum(1)[:, np.newaxis]
            omega = cp.Variable((na), nonneg=True)
            sigma = cp.Variable(1, nonneg=True)
            constraints = [cp.sum(omega) == 1, omega >= sigma]

            def upper_bound_eq_5(s: int):
                obj_supp = []
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * cp.inv_pos(omega[a]) * (2 + 8 * golden_ratio * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    
                    T2 = self.normalizer * cp.inv_pos(omega[pi_greedy[s]]) * (2 + 8 * golden_ratio * Mk[s, pi_greedy[s]]) / (tol + Delta_sq[s,a] * ((1 - self.discount_factor) ** 2))

                    entropy = 1#np.log(ns)+(np.log(self.P) * self.P).sum(-1)[s,a]
                    
                    obj_supp.append((T1 + T2)*entropy)

                objective = cp.max(cp.hstack(obj_supp))
                return objective
            
            problem = cp.Problem(cp.Minimize(upper_bound_eq_5(s)), constraints)
            result = problem.solve(verbose=False, solver=cp.MOSEK)#, abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=200)
            policy[s] = omega.value

        omega = compute_stationary_distribution(policy, self.P)
        return policy, self.evaluate_allocation(omega)

    
    def compute_optimal_soft_policy(self, discount_explorative: float, theta: float):
        assert theta > 0 and theta < np.infty, 'Theta must be in (0, infty)'
        assert discount_explorative > 0 and discount_explorative < 1, 'discount_explorative must be in (0,1)'
        
        # New MDP definition
        Rnew = np.zeros((self.dim_state, self.dim_action))
 
        min_delta = self.delta_sq_min# self.delta_sq[self.idxs_subopt_actions].reshape(self.dim_state, self.dim_action-1).min(-1)[:, np.newaxis]
        Rnew = np.maximum(self.delta_sq, min_delta) / (2 + 8 *golden_ratio * self.Mk_V_greedy)
        Rnew = -(1-discount_explorative)*Rnew[:, :, np.newaxis]
        
        pi = np.ones((self.dim_state, self.dim_action)) / self.dim_action
        
        return soft_policy_iteration(discount_explorative, theta, self.P, Rnew, pi)
        
    
if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ns, na = 5,15
    np.random.seed(2)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    
    discount_factor = 0.99
    
    mdp = MDPDescription(P, R, discount_factor)
    import matplotlib.pyplot as plt
    
    values = np.geomspace(1e-3, 1, 30)
    
    new_mdp = MDPDescription2(P, R, discount_factor, moment_order_k=1)
    
    X = (mdp.var_V_greedy / mdp.span_V_greedy ** (4/3)) * (8/3)
    #mdp.
    import pdb
    pdb.set_trace()
    
    print(new_mdp.compute_optimal_soft_policy(0.99, 1e-3))
    eval_orig = new_mdp.compute_allocation(navigation_constraints=True)[-1]
    
    
    eval = []
    eval2 = []
    for theta in values:
        print(theta)
        pi = new_mdp.compute_optimal_soft_policy(0.1, theta)[1]
        omega_new  = compute_stationary_distribution(pi, P)
        eval.append(new_mdp.evaluate_allocation(omega_new))
        
        pi = new_mdp.compute_optimal_soft_policy(0.99, theta)[1]
        omega_new  = compute_stationary_distribution(pi, P)
        eval2.append(new_mdp.evaluate_allocation(omega_new))
    
    plt.plot(values, eval / eval_orig,label='eval1')
    plt.plot(values, eval2 / eval_orig,label='eval2')
    plt.plot(values, np.ones(len(values)))
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.show()
        
    
    # alloc2, v2 = new_mdp.compute_allocation(type=BoundType.BOUND_1)
    # alloc3, v3 = new_mdp.compute_allocation(type=BoundType.BOUND_1, navigation_constraints=True)
    
    # print(f'{mdp.evaluate_allocation(alloc0)} \t {mdp.evaluate_allocation(alloc2)}')
    # print(f'{new_mdp.evaluate_allocation(alloc0, type=BoundType.BOUND_1)} \t {new_mdp.evaluate_allocation(alloc2, type=BoundType.BOUND_1)}')
    # print(f'{new_mdp.evaluate_allocation(alloc0, type=BoundType.BOUND_2)} \t {new_mdp.evaluate_allocation(alloc2, type=BoundType.BOUND_2)}')
    # print('-------')
    # print(f'{mdp.evaluate_allocation(alloc1, navigation_constraints=True)} \t {mdp.evaluate_allocation(alloc3, navigation_constraints=True)}')
    # print(f'{new_mdp.evaluate_allocation(alloc1, type=BoundType.BOUND_1, navigation_constraints=True)} \t {new_mdp.evaluate_allocation(alloc3, type=BoundType.BOUND_1, navigation_constraints=True)}')
    # print(f'{new_mdp.evaluate_allocation(alloc1, type=BoundType.BOUND_2, navigation_constraints=True)} \t {new_mdp.evaluate_allocation(alloc3, type=BoundType.BOUND_2, navigation_constraints=True)}')
    
    # print('-------')
    # print(f'{mdp.evaluate_allocation(alloc0)} \t {new_mdp.evaluate_allocation(alloc0, type=BoundType.BOUND_1)}')
    # print(f'{mdp.evaluate_allocation(alloc2)} \t {new_mdp.evaluate_allocation(alloc2, type=BoundType.BOUND_1)}')

    # print(f'{mdp.evaluate_allocation(alloc1, navigation_constraints=True)} \t {new_mdp.evaluate_allocation(alloc1, type=BoundType.BOUND_1, navigation_constraints=True)}')
    # print(f'{mdp.evaluate_allocation(alloc3, navigation_constraints=True)} \t {new_mdp.evaluate_allocation(alloc3, type=BoundType.BOUND_1, navigation_constraints=True)}')
    # print(new_mdp.compute_allocation(type=BoundType.BOUND_2))
    # print(new_mdp.compute_allocation(type=BoundType.BOUND_2, navigation_constraints=True))
