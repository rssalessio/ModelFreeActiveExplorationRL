from riverswim import RiverSwim
from new_mdp_description import MDPDescription2, BoundType, MDPDescription
import numpy as np

env = RiverSwim(5)

mdp = MDPDescription2(env.transitions, env.rewards[..., np.newaxis], 0.99, 1)
mdp2 = MDPDescription(env.transitions, env.rewards[..., np.newaxis], 0.99)
allocation1 = mdp.compute_allocation(type=BoundType.BOUND_1)
allocation2 = mdp.compute_allocation(type=BoundType.BOUND_2)
import pdb
pdb.set_trace()