import numpy as np
import numpy.typing as npt
import cvxpy as cp
from BestPolicyIdentification.utils import policy_iteration
from typing import Tuple
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
        
        if type == BoundType.BOUND_1:
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * (2 + 8 * golden_ratio * self.Mk_V_greedy[s,a]) / (omega[s,a] * self.delta_sq[s,a])
                    #T1 = T1/self.normalizer
                    
                    obj_supp_s = []
                    for sp in range(ns):
                        T2 = self.normalizer * (2 + 8 * golden_ratio * self.Mk_V_greedy[sp, self.pi_greedy[sp]]) / (omega[sp, self.pi_greedy[sp]] * self.delta_sq[s,a] * ((1 - discount_factor) ** 2))
                        obj_supp_s.append(T2)# / self.normalizer)
                    
                    obj_supp.append(T1 + np.max(obj_supp_s))

            objective = np.max(obj_supp)
            return objective / self.normalizer
        
        elif type == BoundType.BOUND_2:
            objective = 0
            for s in range(ns):
                obj_supp = []
                
                T2 = self.normalizer *  (2 + 8 * golden_ratio * self.Mk_V_greedy[s, self.pi_greedy[s]]) / (omega[s, self.pi_greedy[s]] * self.delta_sq_min * ((1 - self.discount_factor) ** 2))
                #T2 = T2 / self.normalizer
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * (2 + 8 * golden_ratio * self.Mk_V_greedy[s,a]) / (omega[s,a] * (self.delta_sq[s,a]))
                    obj_supp.append(T1) #/ self.normalizer)
                    
                objective += T2 + np.max(obj_supp)

            return objective / self.normalizer
        else:
            raise Exception(f'Type {type} not found')
        
    
    def compute_allocation(self, type: BoundType = BoundType.BOUND_1, navigation_constraints: bool = False) -> Tuple[npt.NDArray[np.float64], float]:
        omega = cp.Variable((ns, na), nonneg=True)
        sigma = cp.Variable(1, nonneg=True)
        constraints = [cp.sum(omega) == 1, omega >= sigma]#, sigma >= 1e-12]
        
        if navigation_constraints:
            constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)])

        tol = 0#1e-14
        def upper_bound_eq_5():
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * cp.inv_pos(omega[s,a]) * (2 + 8 * golden_ratio * self.Mk_V_greedy[s,a]) / (self.delta_sq[s,a]+tol)
                    #T1 = T1 / self.normalizer
                    
                    obj_supp_s = []
                    for sp in range(ns):
                        T2 = self.normalizer * cp.inv_pos(omega[sp, self.pi_greedy[sp]]) * (2 + 8 * golden_ratio * self.Mk_V_greedy[sp, self.pi_greedy[sp]]) / (tol + self.delta_sq[s,a] * ((1 - self.discount_factor) ** 2))
                        #T2 = T2/ self.normalizer
                        obj_supp_s.append(T2)
                    
                    obj_supp.append(T1 + cp.max(cp.hstack(obj_supp_s)))

            objective = cp.max(cp.hstack(obj_supp))
            return objective

        def upper_bound_eq_6():
            objective = 0
            for s in range(ns):
                obj_supp = []
                
                T2 = self.normalizer * cp.inv_pos(omega[s, self.pi_greedy[s]]) * (2 + 8 * golden_ratio * self.Mk_V_greedy[s, self.pi_greedy[s]]) /  (tol+self.delta_sq_min * ((1 - self.discount_factor) ** 2))
                #T2 = T2 / self.normalizer
                for a in range(na):
                    if a == self.pi_greedy[s]: continue
                    T1 = self.normalizer * cp.inv_pos(omega[s,a] ) * (2 + 8 * golden_ratio * self.Mk_V_greedy[s,a]) / (self.delta_sq[s,a]+tol)
                    #T1 = T1/ self.normalizer
                    obj_supp.append(T1)
                    
                objective += T2 + cp.max(cp.hstack(obj_supp))

            return objective
        
        if type == BoundType.BOUND_1:
            objective = cp.Minimize(upper_bound_eq_5())
        elif type == BoundType.BOUND_2:
            objective = cp.Minimize(upper_bound_eq_6())
        else:
            raise Exception(f'Type {type} not found')

        problem = cp.Problem(objective, constraints)
        result = problem.solve(verbose=False, solver=cp.MOSEK)#, abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=200)

        return omega.value, self.evaluate_allocation(omega.value, type, navigation_constraints=navigation_constraints)
    
    
if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ns, na = 5,3
    np.random.seed(2)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    
    discount_factor = 0.99
    
    mdp = MDPDescription(P, R, discount_factor)
    alloc0, v0 = mdp.compute_allocation()
    alloc1, v1 = mdp.compute_allocation(True)
    
    new_mdp = MDPDescription2(P, R, discount_factor, moment_order_k=1)
    
    alloc2, v2 = new_mdp.compute_allocation(type=BoundType.BOUND_1)
    alloc3, v3 = new_mdp.compute_allocation(type=BoundType.BOUND_1, navigation_constraints=True)
    
    print(f'{mdp.evaluate_allocation(alloc0)} \t {mdp.evaluate_allocation(alloc2)}')
    print(f'{new_mdp.evaluate_allocation(alloc0, type=BoundType.BOUND_1)} \t {new_mdp.evaluate_allocation(alloc2, type=BoundType.BOUND_1)}')
    print(f'{new_mdp.evaluate_allocation(alloc0, type=BoundType.BOUND_2)} \t {new_mdp.evaluate_allocation(alloc2, type=BoundType.BOUND_2)}')
    print('-------')
    print(f'{mdp.evaluate_allocation(alloc1, navigation_constraints=True)} \t {mdp.evaluate_allocation(alloc3, navigation_constraints=True)}')
    print(f'{new_mdp.evaluate_allocation(alloc1, type=BoundType.BOUND_1, navigation_constraints=True)} \t {new_mdp.evaluate_allocation(alloc3, type=BoundType.BOUND_1, navigation_constraints=True)}')
    print(f'{new_mdp.evaluate_allocation(alloc1, type=BoundType.BOUND_2, navigation_constraints=True)} \t {new_mdp.evaluate_allocation(alloc3, type=BoundType.BOUND_2, navigation_constraints=True)}')
    
    print('-------')
    print(f'{mdp.evaluate_allocation(alloc0)} \t {new_mdp.evaluate_allocation(alloc0, type=BoundType.BOUND_1)}')
    print(f'{mdp.evaluate_allocation(alloc2)} \t {new_mdp.evaluate_allocation(alloc2, type=BoundType.BOUND_1)}')

    print(f'{mdp.evaluate_allocation(alloc1, navigation_constraints=True)} \t {new_mdp.evaluate_allocation(alloc1, type=BoundType.BOUND_1, navigation_constraints=True)}')
    print(f'{mdp.evaluate_allocation(alloc3, navigation_constraints=True)} \t {new_mdp.evaluate_allocation(alloc3, type=BoundType.BOUND_1, navigation_constraints=True)}')
    # print(new_mdp.compute_allocation(type=BoundType.BOUND_2))
    # print(new_mdp.compute_allocation(type=BoundType.BOUND_2, navigation_constraints=True))
