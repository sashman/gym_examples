"""CLI interface for gym_examples project.

Be creative! do whatever you want!

- Install click or typer and create a CLI app
- Use builtin argparse
- Start a web application
- Import things from your .base module
"""

from gym_examples.warhammer40k.model.dqn.agent import Agent


def main(hyperparameters: str, train: bool, render: bool):  # pragma: no cover

    dql = Agent(hyperparameter_set=hyperparameters)

    if train:
        dql.run(is_training=True, render=render)
    else:
        dql.run(is_training=False, render=render)