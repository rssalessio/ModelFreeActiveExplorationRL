# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
import numpy.typing as npt
from typing import Tuple

class RandomMDP(object):
    """RandomMDP environment
    """
    ns: int                                 # Number of states
    na: int                                 # Number of actions
    transitions: npt.NDArray[np.float64]    # Transition function P(s'|s,a)
    rewards: npt.NDArray[np.float64]        # Rewards r(s,a,s')
    current_state: int                      # CUrrent state
    
    def __init__(self, 
                 num_states: int = 5,
                 num_actions: int = 3):
        """Initialize a river swim environment

        Parameters
        ----------
        num_states : int, optional
            Number of states, by default 5
        num_actions : float, optional
          Number of actions, by default 3
        """        
        self.ns = num_states
        self.na = num_actions

        self.theta = theta = np.ones(num_states) + np.linspace(0, num_states/10 - 0.1, num_states).cumsum()
        self.transitions = np.random.dirichlet(theta, size=(num_states, num_actions))
        self.rewards = np.random.dirichlet(theta, size=(num_states, num_actions))

        # Reset environment
        self.reset()
    
    def reset(self) -> int:
        """Reset the current state

        Returns
        -------
        int
            initial state 0
        """        
        self.current_state = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float]:
        """Take a step in the environment

        Parameters
        ----------
        action : int

        Returns
        -------
        Tuple[int, float]
            Next state and reward
        """        
        
        next_state = np.random.choice(self.ns, p=self.transitions[self.current_state, action])
        reward = self.rewards[self.current_state, action, next_state]
        self.current_state = next_state
        return next_state, reward
    
