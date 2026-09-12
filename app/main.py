import streamlit as st
# import ollama
# from PIL import Image
# import tempfile
# from pathlib import Path

# # Import your existing pipeline
# from pipeline.inspection_pipeline import inspect_pcb

# st.set_page_config(page_title="AI Defect Command Center", layout="wide")

# st.title("PCB Quality Assurance Dashboard")


# # ============================================================
# # OLLAMA ERROR HELPERS
# # ============================================================

# def show_ollama_error(error):
#     """
#     Display an actionable warning banner inside the Streamlit
#     dashboard instead of crashing with a raw traceback.
#     """
#     status_code = getattr(error, "status_code", None)
#     error_text = getattr(error, "error", str(error))

#     st.error(
#         f"🚧 **Ollama model error** — diagnosis could not be generated.\n\n"
#         f"{error_text}"
#         f"{f' (HTTP status {status_code})' if status_code is not None else ''}\n\n"
#         f"**How to fix this:**\n"
#         f"- Free up GPU/CPU memory by closing other Ollama model sessions.\n"
#         f"- Pull the lightweight fallback model with "
#         f"`ollama pull gemma:2b` if it is not installed.\n"
#         f"- Restart the Ollama server with `ollama serve`.\n"
#         f"- Retry the inspection after the model is available."
#     )


# def show_connection_error(error):
#     """
#     Display an actionable warning banner when the Ollama server
#     cannot be reached.
#     """
#     st.error(
#         f"🚧 **Ollama connection error** — cannot reach the Ollama server.\n\n"
#         f"{getattr(error, 'error', str(error))}\n\n"
#         f"**How to fix this:**\n"
#         f"- Make sure the Ollama server is running with `ollama serve`.\n"
#         f"- Check that Ollama is listening on the expected address "
#         f"(default `http://localhost:11434`).\n"
#         f"- Retry the inspection after the server is reachable."
#     )


# # Sidebar Controls
# st.sidebar.header("Inspection Controls")
# uploaded_file = st.sidebar.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])
# confidence = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.25)

# if uploaded_file is not None:
#     # Save uploaded image temporarily for YOLO to read
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_path = temp_file.name

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Live Optical Feed")
#         st.image(Image.open(temp_path), use_container_width=True)

#     with col2:
#         st.subheader("Gemma 3 Diagnostic Output")
#         with st.spinner("Executing YOLO & RAG Pipeline locally..."):
#             try:
#                 # Run your exact pipeline function
#                 results = inspect_pcb(temp_path, confidence_threshold=confidence)

#                 # Display Final Decision
#                 if results["final_decision"] == "REJECT":
#                     st.error("🚨 FINAL DECISION: REJECT")
#                 else:
#                     st.success("✅ FINAL DECISION: REPAIR")

#                 # Display LLM Reasons
#                 for diag in results["diagnoses"]:
#                     st.markdown(f"**Defect:** {diag['detection']['class_name']}")
#                     st.markdown(f"**Reasoning:** {diag['decision']['reason']}")

#                     # Show a warning banner if a fallback model was used
#                     if diag.get("fallback_used"):
#                         st.warning(diag.get("llm_warning") or "Fallback model was used for this diagnosis.")

#             except ollama.ResponseError as e:
#                 show_ollama_error(e)

#             except ollama.RequestError as e:
#                 show_connection_error(e)



# from rl.robot_env import RobotSortingEnv
# from rl.dqn_agent import DQNAgent

# # ... (Previous YOLO/RAG execution code) ...

# # 1. Determine the target bin based on Gemma's decision
# decision = results.get("final_decision")
# target_bin = 0 if decision == "REPAIR" else 1

# st.markdown("---")
# st.subheader("Autonomous Robotic Sorting")

# with st.spinner("Calculating optimal robot path..."):
#     # 2. Initialize the environment and load your trained agent
#     env = RobotSortingEnv()
#     agent = DQNAgent(state_size=7, action_size=6)
    
#     try:
#         agent.load("rl/models/pcb_sorting_dqn.pt")
#         agent.epsilon = 0.0  # Force optimal pathing
        
#         # 3. Execute the sorting simulation
#         state = env.reset(target_bin=target_bin)
#         path_taken = [env.robot_position]
        
#         for step in range(env.max_steps):
#             action = agent.choose_action(state)
#             state, reward, done, info = env.step(action)
#             path_taken.append(env.robot_position)
            
#             if done:
#                 break
                
#         if info.get("success"):
#             st.success(f"🤖 Robot successfully routed the PCB to the {decision} bin!")
#             st.write(f"**Path Coordinates taken:** {path_taken}")
            
#     except FileNotFoundError:
#         st.warning("⚠️ DQN model not found. Please run `python rl/train_dqn.py` first.")

# import streamlit as st

# st.set_page_config(
#     page_title="Industrial AI Pipeline", 
#     layout="wide"
# )

st.title("🏭 End-to-End Autonomous PCB Factory")
st.markdown("""
Welcome to the AI Control Center. Please use the sidebar to navigate through the pipeline:

1. **Step 1: PCB Inspection** - YOLO11 Vision + Gemma RAG Diagnostics
2. **Step 2: Robotic Sorting** - DQN Reinforcement Learning Pathfinding
""")