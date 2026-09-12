import pybullet as p
import pybullet_data
import numpy as np
import time

def grid_to_world(grid_pos, z_height=0.1):
    """
    Translates a 10x10 DQN grid coordinate to a 3D physical coordinate.
    Grid center (5,5) becomes world origin (0,0).
    """
    x = (grid_pos[0] - 5) * 0.1  # Scale down for robotic workspace
    y = (grid_pos[1] - 5) * 0.1
    return [x, y, z_height]

def generate_simulation_frames(path_taken, target_bin_name):
    """
    Runs a headless PyBullet simulation, moving a KUKA arm along the DQN path,
    and yields RGB camera frames for Streamlit to render.
    """
    # 1. Start PyBullet in headless mode (no GUI window)
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 2. Load Environment (Plane and KUKA Arm)
    planeId = p.loadURDF("plane.urdf")
    
    # Load default KUKA iiwa arm provided by pybullet_data
    kukaStartPos = [0, 0, 0]
    kukaStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    kukaId = p.loadURDF("kuka_iiwa/model.urdf", kukaStartPos, kukaStartOrientation)
    
    num_joints = p.getNumJoints(kukaId)
    kuka_end_effector_idx = 6  # The tip of the arm

    # 3. Create visual props (Bins and PCB)
    # PCB Prop at (4,4)
    pcb_pos = grid_to_world((4, 4), 0.05)
    pcb_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.01], rgbaColor=[0.9, 0.7, 0.1, 1])
    pcb_body = p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=pcb_visual, basePosition=pcb_pos)

    # Bins at (8,8) and (8,1)
    repair_pos = grid_to_world((8, 8), 0.01)
    repair_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], rgbaColor=[0.1, 0.8, 0.2, 0.5])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=repair_visual, basePosition=repair_pos)

    reject_pos = grid_to_world((8, 1), 0.01)
    reject_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], rgbaColor=[0.8, 0.1, 0.1, 0.5])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=reject_visual, basePosition=reject_pos)

    # 4. Setup Synthetic Camera
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.2],
        distance=1.2,
        yaw=45,
        pitch=-35,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
    )

    carrying_pcb = False

    # 5. Execute Path via Inverse Kinematics (IK)
    for step in path_taken:
        target_pos = grid_to_world(step, z_height=0.15)
        
        # Calculate joint angles needed to reach the target position
        joint_poses = p.calculateInverseKinematics(kukaId, kuka_end_effector_idx, target_pos)
        
        # Apply motor controls to joints
        for i in range(num_joints):
            p.setJointMotorControl2(
                bodyIndex=kukaId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                maxVelocity=1.5
            )

        # Step the physics engine forward to let the arm move smoothly
        for _ in range(30):
            p.stepSimulation()
            
            # Logic to "pick up" the PCB
            if step == (4, 4):
                carrying_pcb = True
            
            if carrying_pcb:
                # Snap PCB to the robot's end effector
                ee_state = p.getLinkState(kukaId, kuka_end_effector_idx)
                p.resetBasePositionAndOrientation(pcb_body, ee_state[0], ee_state[1])

        # 6. Render the frame
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=600, 
            height=600,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # PyBullet returns RGBA as a flat 1D array, reshape it for Streamlit/Images
        rgb_array = np.reshape(rgbImg, (height, width, 4))
        rgb_array = rgb_array[:, :, :3]  # Drop Alpha channel
        
        yield rgb_array

    p.disconnect()

if __name__ == "__main__":
    print("Testing headless PyBullet simulation generation...")
    test_path = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)]
    generator = generate_simulation_frames(test_path, "REPAIR")
    frames = list(generator)
    print(f"Generated {len(frames)} frames successfully!")