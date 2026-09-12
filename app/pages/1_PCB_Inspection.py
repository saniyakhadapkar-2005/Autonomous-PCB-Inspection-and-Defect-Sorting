# import sys
# from pathlib import Path
# import tempfile
# import streamlit as st
# from PIL import Image

# # Ensure Python can find your pipeline modules
# root_dir = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(root_dir))

# from pipeline.inspection_pipeline import inspect_pcb

# st.title("👁️ Step 1: AI Defect Command Center")

# uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])
# confidence = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.25)

# if uploaded_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_path = temp_file.name

#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(Image.open(temp_path), use_container_width=True, caption="Live Optical Feed")

#     with col2:
#         with st.spinner("Executing YOLO & RAG Pipeline locally..."):
#             results = inspect_pcb(temp_path, confidence_threshold=confidence)
            
#             decision = results.get("final_decision")
            
#             # 🔴 CRITICAL STEP: Save the decision to global memory for Page 2
#             st.session_state['pcb_decision'] = decision
            
#             if decision == "REJECT":
#                 st.error("🚨 FINAL DECISION: REJECT")
#             else:
#                 st.success("✅ FINAL DECISION: REPAIR")
            
#             for diag in results.get("diagnoses", []):
#                 st.markdown(f"**Defect:** {diag['detection']['class_name']}")
#                 st.markdown(f"**Reasoning:** {diag['decision']['reason']}")

# # --- ADD THIS TO KEEP A RUNNING LOG FOR ANALYTICS ---
#             if 'inspection_history' not in st.session_state:
#                 st.session_state['inspection_history'] = []
            
#             # Save the core data of this inspection
#             for diag in results.get("diagnoses", []):
#                 st.session_state['inspection_history'].append({
#                     "Defect": diag['detection']['class_name'],
#                     "Severity": diag['diagnosis'].get('severity', 'Unknown'),
#                     "Cost_INR": str(diag['diagnosis'].get('estimated_repair_cost_inr', '0')).split()[0], # Grab just the number
#                     "Action": diag['decision'].get('decision', 'UNKNOWN')
#                 })


import sys
from pathlib import Path
import tempfile
import streamlit as st
from PIL import Image, ImageDraw

# Ensure Python can find your pipeline modules
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from pipeline.inspection_pipeline import inspect_pcb

st.title("👁️ Step 1: AI Defect Command Center")

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])
confidence = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.25)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Live Optical Feed")
        # 1. Create a placeholder to hold the image so we can overwrite it later
        image_placeholder = st.empty()
        original_img = Image.open(temp_path).convert("RGB")
        image_placeholder.image(original_img, use_container_width=True, caption="Scanning...")

    with col2:
        st.subheader("Gemma Diagnostic Output")
        with st.spinner("Executing YOLO & RAG Pipeline locally..."):
            
            # Run the inspection
            results = inspect_pcb(temp_path, confidence_threshold=confidence)
            
            # ==========================================
            # VISUAL ANNOTATION: DRAW RED CIRCLES
            # ==========================================
            annotated_img = original_img.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            for det in results.get("detections", []):
                x1, y1, x2, y2 = det['bbox']
                
                # Calculate the center and radius to draw a perfect circle
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                radius = max(x2 - x1, y2 - y1) / 2 + 10  # 10 pixel padding
                
                # Draw the red circle (thickness = 6)
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    outline="red",
                    width=6
                )
                
            # 2. Update the UI placeholder with the newly drawn image!
            image_placeholder.image(annotated_img, use_container_width=True, caption="Defects Highlighted")
            # ==========================================
            
            decision = results.get("final_decision")
            
            # Save the decision to global memory for Page 2
            st.session_state['pcb_decision'] = decision
            
            # Save history for Analytics
            if 'inspection_history' not in st.session_state:
                st.session_state['inspection_history'] = []
            
            for diag in results.get("diagnoses", []):
                st.session_state['inspection_history'].append({
                    "Defect": diag['detection']['class_name'],
                    "Severity": diag['diagnosis'].get('severity', 'Unknown'),
                    "Cost_INR": str(diag['diagnosis'].get('estimated_repair_cost_inr', '0')).split()[0],
                    "Action": diag['decision'].get('decision', 'UNKNOWN')
                })
            
            if decision == "REJECT":
                st.error("🚨 FINAL DECISION: REJECT")
            else:
                st.success("✅ FINAL DECISION: REPAIR")
            
            for diag in results.get("diagnoses", []):
                st.markdown(f"**Defect:** {diag['detection']['class_name']}")
                st.markdown(f"**Reasoning:** {diag['decision']['reason']}")