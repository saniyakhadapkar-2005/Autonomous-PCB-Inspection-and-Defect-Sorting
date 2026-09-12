# import sys
# from pathlib import Path
# import streamlit as st

# root_dir = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(root_dir))

# from rl.robot_env import RobotSortingEnv
# from rl.dqn_agent import DQNAgent

# st.title("🤖 Step 2: Autonomous Robotic Sorting")

# # Check if Page 1 has processed a PCB yet
# if 'pcb_decision' not in st.session_state:
#     st.warning("⚠️ No PCB decision found. Please go to '1 PCB Inspection' and upload an image first.")
# else:
#     decision = st.session_state['pcb_decision']
#     target_bin = 0 if decision == "REPAIR" else 1
    
#     st.info(f"Target Acquired: **{decision} BIN**")
    
#     if st.button("Initialize Robot Pathfinding"):
#         with st.spinner("Calculating optimal DQN trajectory..."):
#             env = RobotSortingEnv()
#             agent = DQNAgent(state_size=7, action_size=6)
            
#             try:
#                 agent.load(str(root_dir / "rl" / "models" / "pcb_sorting_dqn.pt"))
#                 agent.epsilon = 0.0  # Force optimal exploitation
                
#                 state = env.reset(target_bin=target_bin)
#                 path_taken = [env.robot_position]
                
#                 for step in range(env.max_steps):
#                     action = agent.choose_action(state)
#                     state, reward, done, info = env.step(action)
#                     path_taken.append(env.robot_position)
                    
#                     if done:
#                         break
                        
#                 if info.get("success"):
#                     st.success(f"🎉 Robot successfully routed the PCB to the {decision} bin!")
#                     st.write(f"**Optimal Coordinates:** {path_taken}")
                    
#             except Exception as e:
#                 st.error(f"Failed to load DQN model: {e}")
#                 st.warning("Make sure you have run `python rl/train_dqn.py` to generate the `.pt` model file.")

import sys
import time
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

# Ensure Python can find your pipeline modules
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from rl.robot_env import RobotSortingEnv
from rl.dqn_agent import DQNAgent

# ============================================================
# SIMULATION RENDERER
# ============================================================
def draw_factory_floor(robot_pos, picked_up, target_bin_name):
    """Draws a 2D grid representing the physical sorting area."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Grid setup
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Draw Bins
    ax.add_patch(plt.Rectangle((7.5, 7.5), 1, 1, color='#2ecc71', alpha=0.6, label='Repair Bin (8,8)'))
    ax.add_patch(plt.Rectangle((7.5, 0.5), 1, 1, color='#e74c3c', alpha=0.6, label='Reject Bin (8,1)'))
    
    # Draw PCB
    if not picked_up:
        ax.plot(4, 4, marker='s', color='#f1c40f', markersize=18, label='PCB at (4,4)')
        
    # Draw Robot
    robot_color = '#3498db' if not picked_up else '#f39c12'
    label = 'Robot (Empty)' if not picked_up else 'Robot (Carrying PCB)'
    ax.plot(robot_pos[0], robot_pos[1], marker='o', color=robot_color, markersize=22, label=label)
    
    ax.set_title(f"Live Trajectory: Routing to {target_bin_name}", fontsize=14, pad=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    return fig

# ============================================================
# UI LAYOUT
# ============================================================
st.title("🤖 Step 2: Autonomous Robotic Sorting")

if 'pcb_decision' not in st.session_state:
    st.warning("⚠️ No PCB decision found. Please go to '1 PCB Inspection' and upload an image first.")
else:
    decision = st.session_state['pcb_decision']
    target_bin = 0 if decision == "REPAIR" else 1
    
    st.info(f"Target Acquired: **{decision} BIN**")
    
    if st.button("Initialize Robot Pathfinding"):
        
        # 1. Calculate the path in the background
        with st.spinner("Calculating optimal DQN trajectory..."):
            env = RobotSortingEnv()
            agent = DQNAgent(state_size=7, action_size=6)
            
            try:
                agent.load(str(root_dir / "rl" / "models" / "pcb_sorting_dqn.pt"))
                agent.epsilon = 0.0  # Force optimal exploitation
                
                state = env.reset(target_bin=target_bin)
                path_taken = [env.robot_position]
                
                for step in range(env.max_steps):
                    action = agent.choose_action(state)
                    state, reward, done, info = env.step(action)
                    path_taken.append(env.robot_position)
                    if done:
                        break
                        
                success = info.get("success")
                
            except Exception as e:
                st.error(f"Failed to load DQN model: {e}")
                st.stop()

        # 2. Playback the simulation visually
        from rl.pybullet_sim import generate_simulation_frames 
        if success:
                    st.success(f"🎉 Optimal path calculated! Commencing 3D PyBullet simulation...")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Create the UI container for the video feed
                        simulation_placeholder = st.image([])
                    
                    # Stream the frames directly from PyBullet Inverse Kinematics
                    for frame in generate_simulation_frames(path_taken, decision):
                        simulation_placeholder.image(frame, channels="RGB", use_container_width=True)
                        
                    st.write(f"**Final Coordinates Reached:** {path_taken[-1]}")
        # if success:
        #     st.success(f"🎉 Optimal path calculated! Commencing physical sorting...")
            
        #     # Center the plot on the dashboard
        #     col1, col2, col3 = st.columns([1, 2, 1])
        #     with col2:
        #         # Create an empty placeholder that we will overwrite in a loop
        #         simulation_placeholder = st.empty()
                
        #     pcb_picked = False
            
        #     # Animate frame by frame
        #     for pos in path_taken:
        #         if pos == (4, 4):
        #             pcb_picked = True
                    
        #         fig = draw_factory_floor(pos, pcb_picked, decision)
                
        #         # Overwrite the placeholder with the new frame
        #         simulation_placeholder.pyplot(fig)
                
        #         # Close figure to prevent system memory leaks
        #         plt.close(fig)
                
        #         # Control animation speed (0.3 seconds per grid step)
        #         time.sleep(0.3)
                
        #     st.write(f"**Final Coordinates Reached:** {path_taken[-1]}")

# (Keep all your standard imports and the DQN logic up until the success block)

# Add this import at the top of 2_Robotic_Sorting.py


# Replace the old `if success:` block with this:
        