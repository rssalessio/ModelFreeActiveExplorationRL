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
            ) -> Tuple[npt.NDArray[np.float64], float]:

        Mk = self.Mk_V_greedy
        Delta_sq = self.delta_sq
        Delta_sq_min = self.delta_sq_min
        pi_greedy = self.pi_greedy
            
            
        
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
    
    
    @staticmethod
    def compute_mf_allocation(
            discount_factor: float,
            Q: npt.NDArray[np.float64],
            Mk: npt.NDArray[np.float64],
            num_visits: npt.NDArray[np.float64],
            type: BoundType = BoundType.BOUND_1,
            navigation_constraints: bool = False, 
            ) ->  npt.NDArray[np.float64]:
    
        ns, na = Q.shape
        
        pi_greedy = Q.argmax(1)
        Delta_sq = np.clip((Q.max(1)[:, np.newaxis] - Q) ** 2, a_min=1e-9, a_max=None)
        idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(na)] for s in range(ns)])
        Delta_sq_subopt = Delta_sq[idxs_subopt_actions]
        Delta_sq_min =  Delta_sq_subopt.min()

        p_transitions = np.ones((ns, na, ns)) + num_visits
        P = p_transitions / p_transitions.sum(-1, keepdims=True)
        normalizer = (Delta_sq_min * (1 -  discount_factor) ** 3) / (ns * na)
        
        
        omega = cp.Variable((ns, na), nonneg=True)
        sigma = cp.Variable(1, nonneg=True)
        constraints = [cp.sum(omega) == 1, omega >= sigma, sigma >= 1e-15]
        
        if navigation_constraints:
            constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(P[:,:,s], omega)) for s in range(ns)])

        tol = 0#1e-14
        def upper_bound_1():
            obj_supp = []
            for s in range(ns):
                for a in range(na):
                    if a == pi_greedy[s]: continue
                    T1 = normalizer * cp.inv_pos(omega[s,a]) * (2 + 8 * golden_ratio * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    
                    obj_supp_s = []
                    for sp in range(ns):
                        T2 = normalizer * cp.inv_pos(omega[sp, pi_greedy[sp]]) * (2 + 8 * golden_ratio * Mk[sp, pi_greedy[sp]]) / (tol + Delta_sq[s,a] * ((1 - discount_factor) ** 2))

                        obj_supp_s.append(T2)
                    
                    obj_supp.append(T1 + cp.max(cp.hstack(obj_supp_s)))

            objective = cp.max(cp.hstack(obj_supp))
            return objective

        def upper_bound_2():
            obj_supp = []
            obj_supp_s = []
            
            for s in range(ns):
                
                T2 = normalizer * cp.inv_pos(omega[s, pi_greedy[s]]) * (2 + 8 * golden_ratio * Mk[s, pi_greedy[s]]) /  (tol+Delta_sq_min * ((1 - discount_factor) ** 2))
                for a in range(na):
                    if a == pi_greedy[s]: continue
                    T1 = normalizer * cp.inv_pos(omega[s,a] ) * (2 + 8 * golden_ratio * Mk[s,a]) / (Delta_sq[s,a]+tol)
                    obj_supp.append(T1)
                
                obj_supp_s.append(T2)

            return cp.max(cp.hstack(obj_supp)) + cp.max(cp.hstack(obj_supp_s))
        
        
        solver = cp.MOSEK
        for _it in range(10):
            try:
                if type == BoundType.BOUND_1:
                    objective = cp.Minimize(upper_bound_1())
                elif type == BoundType.BOUND_2:
                    objective = cp.Minimize(upper_bound_2())
                else:
                    raise Exception(f'Type {type} not found')

                problem = cp.Problem(objective, constraints)
                result = problem.solve(verbose=False, solver=solver, warm_start=True)#, abstol=1e-10, reltol=1e-10, feastol=1e-10, max_iters=200)
                break
            except Exception as e:
                solver = np.random.choice([cp.MOSEK, cp.ECOS, cp.SCS])
                if _it == 9:
                    raise Exception(f'Cannot solve the UB! {e}')
        return omega.value

