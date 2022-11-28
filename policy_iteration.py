import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple


def policy_evaluation(
        gamma: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        pi: NDArray[np.int64],
        V0: Optional[NDArray[np.float64]] = None,
        tol: float = 1e-6) -> NDArray[np.float64]:
    """Policy evaluation

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi (Optional[NDArray[np.int64]], optional): policy
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        tol (float): Tolerance

    Returns:
        NDArray[np.float64]: Value function
    """
    
    NS, NA = P.shape[:2]
    # Initialize values
    if V0 is None:
        V0 = np.zeros(NS)
    
    V = V0.copy()
    while True:
        Delta = 0
        V_next = np.array([P[s, pi[s]] @ (R[s, pi[s]] + gamma * V) for s in range(NS)])
        Delta = np.max([Delta, np.abs(V_next - V).max()])
        V = V_next
        
        if Delta > tol:
            break
    
    return V
        

def policy_iteration(
        gamma: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        pi0: Optional[NDArray[np.int64]] = None,
        V0: Optional[NDArray[np.float64]] = None,
        tol: float = 1e-6) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Policy iteration

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi0 (Optional[NDArray[np.int64]], optional): Initial policy. Defaults to None.
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        tol (float): tolerance

    Returns:
        NDArray[np.float64]: Optimal value function
        NDArray[np.float64]: Optimal policy
    """
    
    NS, NA = P.shape[:2]

    # Initialize values    
    V = V0 if V0 is not None else np.zeros(NS)
    pi = pi0 if pi0 is not None else np.random.binomial(1, p=0.5, size=(NS))
    next_pi = np.zeros_like(pi)
    policy_stable = False
    
    while not policy_stable:
        policy_stable = True
        V = policy_evaluation(gamma, P, R, pi, V, tol)
        for s in range(NS):
            Qs = [P[s,a] @ (R[s,a] + gamma * V) for a in range(NA)]
            next_pi[s] = np.argmax(Qs)
        
        if np.any(next_pi != pi):
            policy_stable = False
        pi = next_pi
    return V, pi
  
    
def gather(X, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = X.shape[:dim] + X.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(X, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)