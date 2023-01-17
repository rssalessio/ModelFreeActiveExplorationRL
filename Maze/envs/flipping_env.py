import numpy as np
from numpy.typing import NDArray, int
from typing import Tuple

class BitFlippingEnvironment(object):
    n: int                          # Number of bits
    _current_state: NDArray[int]     # Current state
    _target_state: NDArray[int]      # Target state
    observation_space: NDArray[int] # Size of the observation space
    
    def __init__(self, n: int):
        """Initializes the bit flipping environment

        Parameters
        ----------
        n : int
            number of bits
        """        
        self.n = n
        self._current_state = None
        self._target_state = None
        self.observation_space = np.arange(2 ** n)
        
    @property
    def num_actions(self) -> int:
        """Returns the number of actions"""
        return self.n
    
    def map_state(self, state: NDArray[int]) -> int:
        """Map a binary string to an integer

        Parameters
        ----------
        state : NDArray[int]
            array in the form of a sequence of (0,1)-digits

        Returns
        -------
        int
            An integer value representing the state
        """        
        return np.dot(state, 1 << np.arange(state.size))
        
    def reset(self) -> int:
        """Resets the current state and target state

        Returns
        -------
        int
            New initial state
        """          
        while True:
            self._current_state = np.random.binomial(1, p=0.5, size=(self.n))
            self._target_state = np.random.binomial(1, p=0.5, size=(self.n))
            if np.any(self._current_state != self._target_state):
                break
        return self.map_state(self._current_state)
        
    def step(self, action: int) -> Tuple[int, float, bool]:
        """Flips the (n-1)th bit of the current state. Returns 0
        if the new string is equal to the target string, otherwise returns
        -1

        Parameters
        ----------
        action : int
            A value in {0, ..., n-1}

        Returns
        -------
        Tuple[int, float, bool]
            Next state, reward, done
        """        
        assert self._current_state is not None, 'Need to reset the environment before using it'
        assert self._target_state is not None, 'Need to reset the environment before using it'
        assert action < self.n, 'The action needs to be in {0, 1, ..., n-1}'
        
        self._current_state[action] = ~self._current_state[action]
        done = np.all(self._current_state == self._target_state)
        return self.map_state(self._current_state), (done - 1), done
        
        
        