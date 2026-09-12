import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DQN NETWORK
# ============================================================

class DQNNetwork(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, state):

        return self.network(state)


# ============================================================
# DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(
        self,
        state_size=7,
        action_size=6
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        self.batch_size = 64

        self.memory = deque(
            maxlen=10000
        )

        # Main network
        self.policy_net = DQNNetwork(
            state_size,
            action_size
        ).to(DEVICE)

        # Target network
        self.target_net = DQNNetwork(
            state_size,
            action_size
        ).to(DEVICE)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

        self.loss_function = nn.MSELoss()

    # ========================================================
    # STORE EXPERIENCE
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    # ========================================================
    # SELECT ACTION
    # ========================================================

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randrange(
                self.action_size
            )

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_net(
                state_tensor
            )

        return int(
            torch.argmax(q_values).item()
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train_step(self):

        if len(self.memory) < self.batch_size:

            return None

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        states = np.array(
            [item[0] for item in batch],
            dtype=np.float32
        )

        actions = np.array(
            [item[1] for item in batch],
            dtype=np.int64
        )

        rewards = np.array(
            [item[2] for item in batch],
            dtype=np.float32
        )

        next_states = np.array(
            [item[3] for item in batch],
            dtype=np.float32
        )

        dones = np.array(
            [item[4] for item in batch],
            dtype=np.float32
        )

        states_tensor = torch.tensor(
            states,
            dtype=torch.float32,
            device=DEVICE
        )

        actions_tensor = torch.tensor(
            actions,
            dtype=torch.long,
            device=DEVICE
        )

        rewards_tensor = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE
        )

        next_states_tensor = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=DEVICE
        )

        dones_tensor = torch.tensor(
            dones,
            dtype=torch.float32,
            device=DEVICE
        )

        # Current Q values
        current_q_values = self.policy_net(
            states_tensor
        ).gather(
            1,
            actions_tensor.unsqueeze(1)
        ).squeeze(1)

        # Next Q values
        with torch.no_grad():

            next_q_values = self.target_net(
                next_states_tensor
            ).max(
                dim=1
            )[0]

        # Bellman equation
        target_q_values = (
            rewards_tensor
            +
            (1 - dones_tensor)
            * self.gamma
            * next_q_values
        )

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            1.0
        )

        self.optimizer.step()

        return loss.item()

    # ========================================================
    # UPDATE TARGET NETWORK
    # ========================================================

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    # ========================================================
    # EPSILON DECAY
    # ========================================================

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save(self, path):

        torch.save(
            self.policy_net.state_dict(),
            path
        )

        print(
            f"✓ DQN model saved: {path}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load(self, path):

        self.policy_net.load_state_dict(
            torch.load(
                path,
                map_location=DEVICE
            )
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.policy_net.eval()

        print(
            f"✓ DQN model loaded: {path}"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DQN AGENT TEST")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    agent = DQNAgent(
        state_size=7,
        action_size=6
    )

    print("\n✓ DQN network created")

    print("\nNetwork architecture:")

    print(agent.policy_net)

    print("\nInitial epsilon:")
    print(agent.epsilon)

    print("\n" + "=" * 70)
    print("✓ DQN AGENT TEST COMPLETED")
    print("=" * 70)