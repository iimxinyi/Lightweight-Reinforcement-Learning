# Lightweight-Reinforcement-Learning
Aiming to achieve lightweight and transferable reinforcement learning algorithms

### Conda Environment Setup and Required Packages

```shell
conda create --name Lightweight_RL python==3.8.20
```

```shell
pip install gym==0.26.2
pip install pygame==2.6.1
pip install numpy==1.24.4
pip install torch==2.4.1
pip install pettingzoo==1.24.3
```

### Test Environment

For Single Agent Deep Reinforcement Learning (SADRL), we use "CartPole-v1".
URL: https://gymnasium.farama.org/environments/classic_control/cart_pole/

For Multi Agent Deep Reinforcement Learning (MADRK), wu use "PettingZoo-SimpleSpread".
URL: https://pettingzoo.farama.org/environments/mpe/simple_spread/
