# import files
import random
import gymnasium as gym
import flappy_bird_gymnasium
import torch
from dqn import DQN
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
from experience_replay import ReplayMemory

# main program

# defining the device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:    
    device = "cpu"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# steps for the game
# 1. set env
# 2. define DQN
# 3. create the experience reply
# 4. multiple episodes DQN train (setup)
# 5. epsilon- greedy policy + hyperparameters (epsilon decay)
# 6. target network
# 7. train and test
class Agent:
    def __init__(self,param_set):
        self.param_set = param_set
        with open("parameters.yaml", "r") as f:
            all_params_set = yaml.safe_load(f)
            params = all_params_set[param_set]
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        self.replay_memory_size = params["replay_memory_size"]
        self.min_batch_size = params["min_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn  =DQN(num_states, num_actions).to(device)
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            # target network
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps=0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
        else:
            # best policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
        for episode in itertools.count():
            state,_ =  env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            episode_reward = 0
            terminated = False
            while (not terminated and episode_reward < self.reward_threshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action,dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                next_state, reward, terminated, _, _ = env.step(action)
                episode_reward += reward

                # crete tensor for next state and reward
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)
                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1
                state = next_state
            print(f"Episode {episode+1} reward: {episode_reward} epsilon: {epsilon:.4f}")
            if is_training:
                # epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            if episode_reward > best_reward:
                log_msg = f"Best reward: {episode_reward} at episode {episode+1}\n"
                with open(self.LOG_FILE, "a") as f:
                    f.write(log_msg+"\n")
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            if is_training and len(memory) > self.min_batch_size:
                # get sample
                mini_batch = memory.sample(self.min_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # sync the network
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
    def optimize(self,mini_batch, policy_dqn, target_dqn):
        # get the batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)
        # calculate the target Q values
        with torch.no_grad():
            target_q = rewards +(1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]
        # calculate y_pred i.e. Q value from current policy
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(dim=1)).squeeze()
        # compute loss
        loss = self.loss_fn(current_q, target_q)
        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train', action='store_true', help='Training the model')
    args = parser.parse_args()
    dql = Agent(param_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True, render=False)
    else:
        dql.run(is_training=False, render=True)