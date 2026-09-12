from pathlib import Path

from rl.robot_env import RobotSortingEnv
from rl.dqn_agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 1000

TARGET_UPDATE_FREQUENCY = 10

MODEL_FOLDER = Path("rl/models")

MODEL_PATH = MODEL_FOLDER / "pcb_sorting_dqn.pt"


# ============================================================
# TRAIN DQN
# ============================================================

def train():

    print("=" * 70)
    print("DQN PCB ROBOT SORTING TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = RobotSortingEnv()

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = DQNAgent(
        state_size=7,
        action_size=6
    )

    print("\n✓ Environment created")
    print("✓ DQN agent created")

    print("\nTraining episodes:", EPISODES)

    rewards_history = []

    success_count = 0

    # ========================================================
    # EPISODES
    # ========================================================

    for episode in range(1, EPISODES + 1):

        # Alternate between repair and reject
        target_bin = 0 if episode % 2 == 0 else 1

        state = env.reset(
            target_bin=target_bin
        )

        total_reward = 0

        loss_values = []

        # ----------------------------------------------------
        # Episode steps
        # ----------------------------------------------------

        for step in range(env.max_steps):

            action = agent.choose_action(
                state
            )

            next_state, reward, done, info = env.step(
                action
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.train_step()

            if loss is not None:

                loss_values.append(loss)

            state = next_state

            total_reward += reward

            if done:

                if info.get(
                    "success",
                    False
                ):

                    success_count += 1

                break

        # ----------------------------------------------------
        # Decay exploration
        # ----------------------------------------------------

        agent.decay_epsilon()

        # ----------------------------------------------------
        # Update target network
        # ----------------------------------------------------

        if episode % TARGET_UPDATE_FREQUENCY == 0:

            agent.update_target_network()

        rewards_history.append(
            total_reward
        )

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if episode == 1 or episode % 10 == 0:

            avg_reward = sum(
                rewards_history[-10:]
            ) / min(
                10,
                len(rewards_history)
            )

            success_rate = (
                success_count / episode
            ) * 100

            avg_loss = (
                sum(loss_values) /
                len(loss_values)
                if loss_values
                else 0
            )

            target_name = (
                "REPAIR"
                if target_bin == 0
                else "REJECT"
            )

            print(
                f"Episode {episode:3d}/{EPISODES} | "
                f"Target={target_name:6s} | "
                f"Reward={total_reward:7.1f} | "
                f"AvgReward={avg_reward:7.1f} | "
                f"Success={success_rate:6.2f}% | "
                f"Epsilon={agent.epsilon:.3f} | "
                f"Loss={avg_loss:.4f}"
            )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    agent.save(
        str(MODEL_PATH)
    )

    print("\n" + "=" * 70)
    print("✓ DQN TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nModel saved at:\n{MODEL_PATH}"
    )

    print(
        f"\nTraining success rate: "
        f"{(success_count / EPISODES) * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()