import random
import math


class RobotSortingEnv:
    """
    DQN environment for autonomous PCB sorting.

    Robot:
        Starts at (0, 0)

    PCB:
        Located at (4, 4)

    Repair Bin:
        (8, 8)

    Reject Bin:
        (8, 1)

    Task:
        Move to PCB -> PICK -> Move to target bin -> DROP
    """

    GRID_SIZE = 10

    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PICK = 4
    DROP = 5

    ACTION_NAMES = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
        4: "PICK",
        5: "DROP"
    }

    def __init__(self):

        self.robot_start = (0, 0)

        self.pcb_position = (4, 4)

        self.repair_bin = (8, 8)

        self.reject_bin = (8, 1)

        self.max_steps = 80

        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, target_bin=None):

        self.robot_position = self.robot_start

        self.pcb_position = (4, 4)

        self.carrying_pcb = False

        self.steps = 0

        self.done = False

        if target_bin is None:

            self.target_bin = random.choice([0, 1])

        else:

            self.target_bin = target_bin

        return self.get_state()

    # ========================================================
    # TARGET POSITION
    # ========================================================

    def get_target_position(self):

        if self.target_bin == 0:

            return self.repair_bin

        return self.reject_bin

    # ========================================================
    # MANHATTAN DISTANCE
    # ========================================================

    def distance(self, position1, position2):

        return abs(
            position1[0] - position2[0]
        ) + abs(
            position1[1] - position2[1]
        )

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self):

        rx, ry = self.robot_position

        px, py = self.pcb_position

        tx, ty = self.get_target_position()

        state = [

            rx / self.GRID_SIZE,
            ry / self.GRID_SIZE,

            px / self.GRID_SIZE,
            py / self.GRID_SIZE,

            tx / self.GRID_SIZE,
            ty / self.GRID_SIZE,

            1.0 if self.carrying_pcb else 0.0
        ]

        return state

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        if self.done:

            return self.get_state(), 0, True, {}

        self.steps += 1

        old_position = self.robot_position

        # ----------------------------------------------------
        # Decide current objective
        # ----------------------------------------------------

        if self.carrying_pcb:

            objective = self.get_target_position()

        else:

            objective = self.pcb_position

        old_distance = self.distance(
            old_position,
            objective
        )

        reward = -0.2

        info = {}

        # ====================================================
        # MOVEMENT
        # ====================================================

        x, y = self.robot_position

        new_x = x

        new_y = y

        if action == self.UP:

            new_y += 1

        elif action == self.DOWN:

            new_y -= 1

        elif action == self.LEFT:

            new_x -= 1

        elif action == self.RIGHT:

            new_x += 1

        # ----------------------------------------------------
        # Boundary
        # ----------------------------------------------------

        if (
            0 <= new_x < self.GRID_SIZE
            and
            0 <= new_y < self.GRID_SIZE
        ):

            self.robot_position = (
                new_x,
                new_y
            )

            new_distance = self.distance(
                self.robot_position,
                objective
            )

            # Reward for moving closer
            distance_change = (
                old_distance - new_distance
            )

            reward += (
                distance_change * 2.0
            )

        else:

            reward = -3.0

        # ====================================================
        # PICK PCB
        # ====================================================

        if action == self.PICK:

            if (
                self.robot_position == self.pcb_position
                and
                not self.carrying_pcb
            ):

                self.carrying_pcb = True

                reward = 30.0

                info["picked"] = True

            else:

                reward = -3.0

        # ====================================================
        # DROP PCB
        # ====================================================

        if action == self.DROP:

            target_position = self.get_target_position()

            if (
                self.robot_position == target_position
                and
                self.carrying_pcb
            ):

                self.carrying_pcb = False

                reward = 100.0

                self.done = True

                info["success"] = True

                if self.target_bin == 0:

                    info["target_bin"] = "REPAIR"

                else:

                    info["target_bin"] = "REJECT"

            else:

                reward = -3.0

        # ====================================================
        # MAX STEPS
        # ====================================================

        if self.steps >= self.max_steps:

            self.done = True

            info["success"] = False

        return (
            self.get_state(),
            reward,
            self.done,
            info
        )

    # ========================================================
    # ACTION NAME
    # ========================================================

    @staticmethod
    def action_name(action):

        return RobotSortingEnv.ACTION_NAMES.get(
            action,
            "UNKNOWN"
        )


# ============================================================
# ENVIRONMENT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ROBOT SORTING ENVIRONMENT TEST")
    print("=" * 70)

    env = RobotSortingEnv()

    state = env.reset(
        target_bin=0
    )

    print("\nTarget Bin: REPAIR")

    print(
        "\nRobot Position:",
        env.robot_position
    )

    print(
        "PCB Position:",
        env.pcb_position
    )

    print(
        "Repair Bin:",
        env.repair_bin
    )

    print(
        "Reject Bin:",
        env.reject_bin
    )

    print(
        "\nInitial State:",
        state
    )

    print("\nAvailable Actions:")

    for action, name in env.ACTION_NAMES.items():

        print(
            f"{action} -> {name}"
        )

    print("\nTesting movement...")

    next_state, reward, done, info = env.step(
        env.RIGHT
    )

    print(
        "\nRobot Position:",
        env.robot_position
    )

    print(
        "Reward:",
        reward
    )

    print(
        "Done:",
        done
    )

    print(
        "Info:",
        info
    )

    print("\n" + "=" * 70)
    print("✓ ROBOT ENVIRONMENT TEST COMPLETED")
    print("=" * 70)