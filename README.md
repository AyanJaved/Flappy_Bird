# 🐦 Flappy Bird DQN Agent

Trained an AI agent to play Flappy Bird using Deep Q-Learning (DQN) with experience replay and a target network — built from scratch using PyTorch and Gymnasium.

---

## 🧠 How It Works

The agent learns to play Flappy Bird by interacting with the environment and updating a neural network (Q-network) to predict the best action in any given state.

Key concepts used:
- **Deep Q-Network (DQN)** — A 2-layer neural network that maps states to Q-values
- **Epsilon-Greedy Policy** — Balances exploration vs exploitation with decaying epsilon
- **Experience Replay** — Stores past transitions and samples random mini-batches to break correlation
- **Target Network** — A separate network updated periodically to stabilize training

---

## 📁 Project Structure

```
├── agent.py               # Main training and testing logic
├── dqn.py                 # Neural network architecture
├── experience_replay.py   # Replay memory buffer
├── parameters.yaml        # Hyperparameters
└── runs/                  # Saved models and logs (auto-generated)
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate (alpha) | 0.001 |
| Discount Factor (gamma) | 0.99 |
| Epsilon Init | 1.0 |
| Epsilon Min | 0.05 |
| Epsilon Decay | 0.9995 |
| Replay Memory Size | 100,000 |
| Batch Size | 32 |
| Network Sync Rate | 10 steps |
| Reward Threshold | 1000 |

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/flappy-bird-dqn.git
cd flappy-bird-dqn

# Install dependencies
pip install gymnasium torch flappy-bird-gymnasium
```

---

## 🚀 Running the Agent

### Training Mode
Trains the agent from scratch and saves the best model to `runs/flappybirdv0.pt`.

```bash
python agent.py flappybirdv0 --train
```

The agent will print episode rewards and epsilon values as it trains:
```
Episode 1 reward: -8.1 epsilon: 1.0000
Episode 2 reward: -7.5 epsilon: 0.9995
...
```

---

### Testing Mode
Loads the saved model and renders the agent playing the game visually.

```bash
python agent.py flappybirdv0
```

> Make sure `runs/flappybirdv0.pt` exists before running test mode (i.e. train first).

---

## 📊 Training Progress

Logs of best rewards are saved to `runs/flappybirdv0.log` during training. Each time the agent beats its previous best, the model is saved automatically.

---

## 📦 Dependencies

- [PyTorch](https://pytorch.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
- PyYAML

---

## 📌 Notes

- Training runs indefinitely (`itertools.count()`), so stop it manually with `Ctrl+C` once rewards are satisfactory.
- GPU is used automatically if available (CUDA or MPS for Apple Silicon).
