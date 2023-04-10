import logging

from gym import envs
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Register openai's environments as multi agent
# This should be done before registering new environments
env_specs = [env_spec for env_spec in envs.registry.all() if 'gym.envs' in env_spec.entry_point]
for spec in env_specs:
    register(
        id='ma_' + spec.id,
        entry_point='ma_gym.envs.openai:MultiAgentWrapper',
        kwargs={'name': spec.id, **spec._kwargs}
    )


register(
    id='minigridTree-v0',
    entry_point='ma_gym.envs.minigrid:MinigridTree'
)

register(
    id='minigridRock-v0',
    entry_point='ma_gym.envs.minigrid:MinigridRock'
)

for game_info in [[(20, 20), 2, 30], [(7, 7), 4, 2]]:  # [(grid_shape, predator_n, prey_n),..]
    grid_shape, n_agents, n_preys = game_info
    _game_name = 'PredatorPrey{}x{}'.format(grid_shape[0], grid_shape[1])
    register(
        id='{}-v0'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys
        }
    )
    # fully -observable ( each agent sees observation of other agents)
    register(
        id='{}-v1'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True
        }
    )

    # prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v2'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys,
            'prey_move_probs': [0, 0, 0, 0, 1]
        }
    )

    # full observability + prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v3'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True,
            'prey_move_probs': [0, 0, 0, 0, 1]
        }
    )
