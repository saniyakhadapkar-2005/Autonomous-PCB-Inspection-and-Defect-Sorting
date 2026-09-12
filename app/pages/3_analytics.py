import streamlit as st
import pandas as pd
import plotly.express as px
import random

st.set_page_config(page_title="Factory Analytics", layout="wide")
st.title("📈 Factory Floor Analytics")
st.markdown("Live tracking of AI diagnostics, defect distributions, and repair costs.")

# ==========================================
# DATA LOADING & MOCK DATA GENERATOR
# ==========================================
# For demonstration/competition purposes, let's inject some dummy data 
# if the live history is empty or too small to look good on a chart.
if 'inspection_history' not in st.session_state or len(st.session_state['inspection_history']) < 5:
    st.info("Injecting simulated historical data to demonstrate dashboard visuals...")
    mock_defects = ["Solder Bridge", "Missing Component", "Oxidation", "Short Circuit"]
    mock_severities = ["Low", "Medium", "High", "Critical"]
    
    mock_data = []
    for _ in range(45):
        defect = random.choice(mock_defects)
        action = "REJECT" if defect == "Short Circuit" or random.random() < 0.2 else "REPAIR"
        mock_data.append({
            "Defect": defect,
            "Severity": random.choice(mock_severities),
            "Cost_INR": random.randint(50, 500) if action == "REPAIR" else 0,
            "Action": action
        })
    df = pd.DataFrame(mock_data)
else:
    # Use real live data from Page 1
    df = pd.DataFrame(st.session_state['inspection_history'])

# Clean up the cost column (ensure it's numeric)
df['Cost_INR'] = pd.to_numeric(df['Cost_INR'], errors='coerce').fillna(0)

# ==========================================
# TOP KPI METRICS
# ==========================================
st.markdown("---")
total_inspected = len(df)
total_repair_cost = df['Cost_INR'].sum()
repair_rate = (len(df[df['Action'] == 'REPAIR']) / total_inspected) * 100 if total_inspected > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total PCBs Inspected", total_inspected)
col2.metric("Cumulative Repair Cost", f"₹{total_repair_cost:,.2f}")
col3.metric("Factory Yield (Repairable)", f"{repair_rate:.1f}%")

st.markdown("---")

# ==========================================
# VISUAL STORYTELLING CHARTS
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Defect Distribution")
    # A clean donut chart showing what types of errors are most common
    defect_counts = df['Defect'].value_counts().reset_index()
    defect_counts.columns = ['Defect', 'Count']
    fig_pie = px.pie(
        defect_counts, 
        names='Defect', 
        values='Count', 
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Routing Decisions")
    # A bar chart showing how many went to the Repair Bin vs Reject Bin
    action_counts = df['Action'].value_counts().reset_index()
    action_counts.columns = ['Action', 'Count']
    fig_bar = px.bar(
        action_counts, 
        x='Action', 
        y='Count',
        color='Action',
        color_discrete_map={"REPAIR": "#2ecc71", "REJECT": "#e74c3c"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ==========================================
# COST ANALYSIS SPREADSHEET
# ==========================================
st.subheader("Raw Financial Impact Log")
st.dataframe(
    df, 
    use_container_width=True,
    column_config={
        "Cost_INR": st.column_config.NumberColumn("Estimated Cost (₹)", format="₹%d")
    }
)