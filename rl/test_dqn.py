from rl.robot_env import RobotSortingEnv
from rl.dqn_agent import DQNAgent

MODEL_PATH = "rl/models/pcb_sorting_dqn.pt"


def run_test(target_bin):

    env = RobotSortingEnv()

    agent = DQNAgent(
        state_size=7,
        action_size=6
    )

    agent.load(MODEL_PATH)

    # Testing time: no random actions
    agent.epsilon = 0.0

    state = env.reset(target_bin=target_bin)

    target_name = "REPAIR" if target_bin == 0 else "REJECT"

    print("\n" + "=" * 70)
    print(f"DQN ROBOT TEST → {target_name} BIN")
    print("=" * 70)

    print(f"\nRobot Start : {env.robot_position}")
    print(f"PCB Position: {env.pcb_position}")
    print(f"Target      : {env.get_target_position()}")

    total_reward = 0
    info = {}

    for step in range(1, env.max_steps + 1):

        action = agent.choose_action(state)

        action_name = env.action_name(action)

        next_state, reward, done, info = env.step(action)

        total_reward += reward

        print(
            f"Step {step:3d} | "
            f"Action={action_name:5s} | "
            f"Position={env.robot_position} | "
            f"Carrying={env.carrying_pcb} | "
            f"Reward={reward:6.1f}"
        )

        state = next_state

        if done:
            break

    print("\n" + "-" * 70)

    if info.get("success", False):
        print("✓ ROBOT SORTING SUCCESS")
        print(f"Target Bin: {info['target_bin']}")
    else:
        print("❌ ROBOT SORTING FAILED")

    print(f"Total Reward: {total_reward}")

    print("=" * 70)


if __name__ == "__main__":

    # Test Repair Bin
    run_test(target_bin=0)

    # Test Reject Bin
    run_test(target_bin=1)