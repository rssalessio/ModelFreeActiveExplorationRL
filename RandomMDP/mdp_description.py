import numpy as np
import numpy.typing as npt
import cvxpy as cp
from utils import policy_iteration
from typing import Tuple

class MDPDescription(object):
    P: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]
    discount_factor: float
    abs_tol: float
    
    V_greedy: npt.NDArray[np.float64]
    pi_greedy: npt.NDArray[np.float64]
    Q_greedy: npt.NDArray[np.float64]
    
    idxs_subopt_actions: npt.NDArray[np.bool_]
    delta_sq: npt.NDArray[np.float64]
    delta_sq_subopt: npt.NDArray[np.float64]
    delta_sq_min: float
    
    avg_V_greedy: npt.NDArray[np.float64]
    var_V_greedy: npt.NDArray[np.float64]
    var_V_greedy_max: float
    span_V_greedy: npt.NDArray[np.float64]
    span_V_greedy_max: float
    
    T1: npt.NDArray[np.float64]
    T2: npt.NDArray[np.float64]
    T3: npt.NDArray[np.float64]
    T4: npt.NDArray[np.float64]
    H: npt.NDArray[np.float64]
    Hstar: npt.NDArray[np.float64]
    
    normalizer: float
    
    
    def __init__(self, P: npt.NDArray[np.float64], R: npt.NDArray[np.float64], discount_factor: float, abs_tol: float = 1e-6):
        self.P = P
        self.R = R
        self.discount_factor = discount_factor
        self.abs_tol = abs_tol

        Rmax, Rmin = np.max(R), np.min(R)
        
        # Normalize rewards
        if Rmax > 1 or Rmin < 0:
            self.R = (self.R - Rmin) / (Rmax - Rmin)

        # Policy iteration
        V, pi, Q = policy_iteration(self.discount_factor, self.P, self.R, atol=self.abs_tol)
        self.V_greedy = V
        self.pi_greedy = pi
        self.Q_greedy = Q
        self.idxs_subopt_actions = np.array([
            [False if self.pi_greedy[s] == a else True for a in range(self.dim_action)] for s in range(self.dim_state)])

        # Compute Delta
        self.delta_sq = np.clip((self.V_greedy[:, np.newaxis] - self.Q_greedy) ** 2, a_min=1e-32, a_max=None)
        self.delta_sq_subopt = self.delta_sq[self.idxs_subopt_actions]
        self.delta_sq_min =  self.delta_sq_subopt.min()
        
        # Compute variance of V, VarMax and Span
        self.avg_V_greedy = self.P @ self.V_greedy
        self.var_V_greedy =  self.P @ (V ** 2) - (self.avg_V_greedy) ** 2
        self.var_V_greedy_max = np.max(self.var_V_greedy[~self.idxs_subopt_actions])
        
        self.span_V_greedy = np.maximum(np.max(self.V_greedy) - self.avg_V_greedy, self.avg_V_greedy- np.min(self.V_greedy))
        self.span_V_greedy_max = np.max(self.span_V_greedy[~self.idxs_subopt_actions])

        # Compute T terms
        self.T1 = np.zeros((self.dim_state, self.dim_action))
        T2_1 = np.zeros_like(self.T1)
        T2_2 = np.zeros_like(self.T1)
        self.T1[self.idxs_subopt_actions] = np.nan_to_num(2 / self.delta_sq_subopt, nan=0, posinf=0, neginf=0)
        T2_1[self.idxs_subopt_actions] = np.nan_to_num(16 * self.var_V_greedy[self.idxs_subopt_actions] / self.delta_sq_subopt, nan=0, posinf=0, neginf=0)
        T2_2[self.idxs_subopt_actions] = np.nan_to_num(6 * self.span_V_greedy[self.idxs_subopt_actions] ** (4/3) / self.delta_sq_subopt ** 2/3, nan=0, posinf=0, neginf=0)
        self.T2 = np.maximum(T2_1, T2_2)
        
        self.T3 = np.nan_to_num(2 / (self.delta_sq_min * ((1 -  discount_factor) ** 2)), nan=0, posinf=0, neginf=0)
        
        self.T4 = np.nan_to_num(min(
            max(
                27 / (self.delta_sq_min * (1 -  discount_factor) ** 3),
                8 / (self.delta_sq_min * ((1-discount_factor) ** 2.5 )),
                14 * (self.span_V_greedy_max/((self.delta_sq_min ** 2/3) * ((1 - discount_factor) ** (4/3))))
            ),
            max(
                16 * self.var_V_greedy_max /  (self.delta_sq_min * (1 - discount_factor)**2),
                6 * (self.span_V_greedy_max/ ((self.delta_sq_min ** 2/3) * ((1-discount_factor) ** (4/3))))
            )
        ), nan=0, posinf=0, neginf=0)
    
        # Compute H and Hstar
        self.H = self.T1 + self.T2
        self.Hstar = self.dim_state * (self.T3 + self.T4)
        
        self.normalizer = (self.delta_sq_min * (1 -  discount_factor) ** 3) / (self.dim_state * self.dim_action)
        
    def _optimal_generative_allocation(self) -> npt.NDArray[np.float64]:
         # Compute allocation vector
        omega = np.copy(self.H)
        if np.isclose(self.H.sum(), 0, atol=0):
            omega = np.ones((self.dim_state, self.dim_action)) / (self.dim_state * self.dim_action)
        else:
            omega[~self.idxs_subopt_actions] = np.sqrt(self.H.sum() * self.Hstar) / self.dim_state
            omega = omega / omega.sum()
        return omega
    
    def _optimal_allocation_with_navigation_constraints(self) -> npt.NDArray[np.float64]:
        H = self.H * self.normalizer
        Hstar = (self.T3 + self.T4)  * self.normalizer
        
        
        omega = cp.Variable((self.dim_state, self.dim_action))
        sigma = cp.Variable(1, nonneg=True)
        constraints = [cp.sum(omega) == 1, omega >= sigma]
        constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)])
        
        objective = cp.max(
            cp.multiply(
                H[self.idxs_subopt_actions].reshape(self.dim_state, self.dim_action-1),
                cp.inv_pos(cp.reshape(omega[self.idxs_subopt_actions], (self.dim_state, self.dim_action-1)))
                )
            ) \
            + cp.max(cp.inv_pos(omega[~self.idxs_subopt_actions]) * Hstar)
        

        objective = cp.Minimize(objective)
        problem = cp.Problem(objective, constraints)
        result = problem.solve(verbose=False, solver=cp.MOSEK)
        omega = omega.value
        res = result * self.normalizer

        return omega

    
    def evaluate_allocation(self, omega: npt.NDArray[np.float64], navigation_constraints: bool = False) -> float:
        # In eq (10) in http://proceedings.mlr.press/v139/marjani21a/marjani21a.pdf
        # the authors claim that is 2*(sum(H) + Hstar), however, from results it seems like
        # it's just (sum(H) + Hstar)
        # _U1 = 2 * (np.sum(H) + Hstar)
        
        # This comes from the code of the original paper, even though corollary 1 has not the following form
        # _U2 = (H.sum() + Hstar + 2*np.sqrt(H.sum() * Hstar) )
        if navigation_constraints is True:
            checks = np.abs(np.array([np.sum(omega[s]) - np.sum(np.multiply(self.P[:,:,s], omega)) for s in range(self.dim_state)]))
            assert checks.min() < 1e-5, "Allocation does not satisfy navigation constraints"
        
        H = self.H / self.normalizer
        Hstar = (self.T3 + self.T4) / self.normalizer
        U = np.max(
            H[self.idxs_subopt_actions].reshape(self.dim_state, self.dim_action-1)/omega[self.idxs_subopt_actions].reshape(self.dim_state, self.dim_action-1)) \
                + np.max(Hstar/ (omega[~self.idxs_subopt_actions]))
        return U * self.normalizer

    def compute_allocation(self, navigation_constraints: bool = False) -> Tuple[npt.NDArray[np.float64], float]:
        if navigation_constraints is False:
            omega = self._optimal_generative_allocation()
        else:
            omega = self._optimal_allocation_with_navigation_constraints()
        return omega, self.evaluate_allocation(omega, navigation_constraints=navigation_constraints)
        
    @property
    def dim_state(self) -> int:
        return self.P.shape[0]
    
    @property
    def dim_action(self) -> int:
        return self.P.shape[1]
    
if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ns, na = 5,3
    np.random.seed(2)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    
    discount_factor = 0.99
    
    mdp = MDPDescription(P, R, discount_factor)
    
    print(mdp.compute_allocation())
    print(mdp.compute_allocation(True))
