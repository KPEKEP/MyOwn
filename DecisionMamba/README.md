# Decision Mamba for CartPole-v1

## Project Overview

This project is an implementation of a Decision Mamba, a novel architecture that combines reinforcement learning and transformer models, 
to solve the CartPole-v1 environment from OpenAI Gym. The Decision Mamba is inspired by Decision Transformer architecture, but it uses Mamba instead of transformer to learn past experiences to make decisions, predicting the next best action 
based on the current state and the desired return.

[Decision Transformer paper](https://arxiv.org/pdf/2106.01345.pdf)
[Mamba paper](https://arxiv.org/abs/2312.00752)
## Features

- **Environment**: Utilizes OpenAI Gym's CartPole-v1 environment.
- **Policy**: Implements a custom theta-omega policy for action decision. As described here: https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f
- **Dataset**: Custom dataset generation from the environment for training the model.
- **Training and Evaluation**: Functions for training the model and evaluating its performance in the environment.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Pavel Nakaznenko, 2024

## Acknowledgements
- [Simplified annotated Mamba implementation](https://github.com/KPEKEP/MyOwn/Mamba)
- [Mamba paper](https://arxiv.org/abs/2312.00752)
- [OpenAI Gym](https://gym.openai.com/)
- [PyTorch](https://pytorch.org/)
- [Jian Xu](https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f)
- [Original authors](https://arxiv.org/pdf/2106.01345.pdf)
- [Reference Decision Transformer implementation](https://github.com/kzl/decision-transformer)