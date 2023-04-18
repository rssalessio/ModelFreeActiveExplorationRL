import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn
from bsuite.baselines.tf import boot_dqn
from bsuite import sweep
from explorative_agent import default_agent as explorative_default_agent
from onpolicy_agent import default_agent as onpolicy_default_agent
from gen_proj import default_agent as MFBPIProjected_default_agent
from boot_explorative import default_agent as BootExplorative_default_agent
from agents.boot_dqn import default_agent as boot_dqn_torch_default_agent
SAVE_PATH_DQN = './logs/explorative_agent'


agents = (
    #(BootExplorative_default_agent, './logs/boot_explorative'),
    #(MFBPIProjected_default_agent, './logs/mfbpi_projected'),
    #(onpolicy_default_agent, './logs/onpolicy_agent'),
    (explorative_default_agent, './logs/explorative_agent'),
    #(dqn.default_agent, './logs/dqn'),
    #(boot_dqn_torch_default_agent, './logs/boot_dqn_torch'),
    #(boot_dqn.default_agent, './logs/boot_dqn'),
)

        
# for bsuite_id in sweep.CARTPOLE_NOISE[:3]:
#     print('---------------------')
#     print(bsuite_id)
#     for make_agent, path in agents:
#         print(path)
#         env = bsuite.load_and_record(bsuite_id, save_path=path, overwrite=True)
#         agent = make_agent(
#             obs_spec=env.observation_spec(),
#             action_spec=env.action_spec()
#         )
#         experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)

# for bsuite_id in sweep.CARTPOLE_SWINGUP[:3]:
#     print('---------------------')
#     print(bsuite_id)
#     for make_agent, path in agents:
#         print(path)
#         env = bsuite.load_and_record(bsuite_id, save_path=path, overwrite=True)
#         agent = make_agent(
#             obs_spec=env.observation_spec(),
#             action_spec=env.action_spec()
#         )
#         experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)
        
for bsuite_id in sweep.DEEP_SEA:
    print('---------------------')
    print(bsuite_id)
    for make_agent, path in agents:
        print(path)
        env = bsuite.load_and_record(bsuite_id, save_path=path, overwrite=True)
        agent = make_agent(
            obs_spec=env.observation_spec(),
            action_spec=env.action_spec()
        )
        experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)
    exit(-1)