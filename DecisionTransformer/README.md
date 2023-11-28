# Decision Transformer for CartPole-v1

## Project Overview

This project is an implementation of a Decision Transformer, an architecture that combines reinforcement learning and transformer models, 
to solve the CartPole-v1 environment from OpenAI Gym. The Decision Transformer uses past experiences to make decisions, predicting the next best action 
based on the current state and the desired return.

Paper: https://arxiv.org/pdf/2106.01345.pdf

## Features

- **Environment**: Utilizes OpenAI Gym's CartPole-v1 environment.
- **Policy**: Implements a custom theta-omega policy for action decision. As described here: https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f
- **Dataset**: Custom dataset generation from the environment for training the model.
- **Model**: Decision Transformer model with customizable layers, embedding dimensions, and other hyperparameters.
- **Training and Evaluation**: Functions for training the model and evaluating its performance in the environment.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Pavel Nakaznenko, 2023

## Acknowledgements

- [OpenAI Gym](https://gym.openai.com/)
- [PyTorch](https://pytorch.org/)
- [Jian Xu](https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f)
- [Original authors](https://arxiv.org/pdf/2106.01345.pdf)
- [Reference implementation](https://github.com/kzl/decision-transformer)