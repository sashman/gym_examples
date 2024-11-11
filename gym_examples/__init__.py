from gymnasium.envs.registration import register

register(
    id="gymnasium_env/Warhammer40k-v0",
    entry_point="gym_examples.warhammer40k.envs.warhammer40k:Warhammer40kEnv",
)