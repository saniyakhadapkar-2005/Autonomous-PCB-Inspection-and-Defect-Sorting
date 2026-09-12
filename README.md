# 🔍 Autonomous PCB Inspection and Defect Sorting

An AI-powered system for **automated PCB defect detection, repairability analysis, robotic sorting, and final report generation** using Computer Vision, Retrieval-Augmented Generation (RAG), and Reinforcement Learning.

The system analyzes PCB images, detects manufacturing defects using **YOLO**, retrieves relevant repair knowledge using **RAG**, determines whether the PCB is repairable or rejectable, performs robotic sorting using a **DQN agent with PyBullet simulation**, and generates a final PCB inspection report.

---

## 🚀 Project Overview

Manual PCB inspection is time-consuming, repetitive, and prone to human error.

This project provides an intelligent automated pipeline that combines:

* 👁️ **Computer Vision** for PCB defect detection
* 🧠 **RAG** for repair knowledge retrieval
* ⚙️ **Decision Engine** for repairability analysis
* 🤖 **Deep Reinforcement Learning** for robotic sorting
* 🦾 **PyBullet** for robotic-arm simulation
* 📄 **Automated Report Generation** for final PCB inspection results
* 📊 **Streamlit** for an interactive application

---

## 🔄 End-to-End Workflow

```text
PCB Image
    ↓
YOLO Defect Detection
    ↓
Defect Classification
    ↓
RAG Knowledge Retrieval
    ↓
Repairability Analysis
    ↓
Decision Engine
    ↓
┌───────────────────────┐
│                       │
↓                       ↓
Repairable          Non-Repairable
│                       │
↓                       ↓
Repair Bin          Reject Bin
│                       │
└──────────┬────────────┘
           ↓
   DQN Robotic Agent
           ↓
    PyBullet Simulation
           ↓
      Sorting Action
           ↓
 Final PCB Report Generation
```

---

## 🎯 Problem Statement

PCB manufacturing can generate defective boards such as:

* Damaged Board
* Damaged Component
* Missing Component
* Solder Bridge
* Solder Ball

Traditional inspection requires human operators to manually identify defects and decide whether a PCB should be repaired or rejected.

### Project Goals

1. Detect PCB defects automatically.
2. Identify the type of defect.
3. Retrieve relevant repair information.
4. Determine whether the defect is repairable.
5. Decide between repair and reject.
6. Simulate robotic sorting using Reinforcement Learning.
7. Generate a final PCB inspection and sorting report.

---

## ✨ Key Features

* 🔍 **Automated PCB Defect Detection** using YOLO
* 🎯 **Defect Classification and Localization**
* 🧠 **RAG-based Repair Knowledge Retrieval**
* ⚙️ **Repair / Reject Decision Engine**
* 🤖 **DQN-based Robotic Sorting**
* 🦾 **PyBullet Robotic Simulation**
* 📊 **Interactive Streamlit Dashboard**
* 📄 **Final PCB Inspection Report Generation**
* 🧩 **Modular Python Architecture**

---

## 📄 Final PCB Report

After inspection and sorting, the system generates a final report containing relevant PCB analysis information such as:

* PCB inspection result
* Detected defect(s)
* Defect classification
* Repairability decision
* Recommended action
* Sorting result
* Final inspection summary

### Report Flow

```text
Inspection
    ↓
Defect Detection
    ↓
Repairability Analysis
    ↓
Sorting
    ↓
Final PCB Report
```

---

## 🧰 Technology Stack

| Area                   | Technology                    |
| ---------------------- | ----------------------------- |
| Programming            | Python                        |
| Computer Vision        | OpenCV                        |
| Object Detection       | YOLO                          |
| Deep Learning          | PyTorch                       |
| NLP / RAG              | Embeddings + Vector Retrieval |
| Knowledge Base         | PCB Repair Knowledge          |
| Reinforcement Learning | DQN                           |
| Robotics Simulation    | PyBullet                      |
| RL Environment         | Gymnasium                     |
| Web UI                 | Streamlit                     |
| Configuration          | YAML                          |
| Version Control        | Git & GitHub                  |

---

## 📁 Project Structure

```text
Autonomous-PCB-Inspection-and-Defect-Sorting/
│
├── app/
│   ├── components/
│   ├── pages/
│   └── main.py
│
├── config/
│
├── knowledge_base/
│   └── pcb_repair_knowledge.txt
│
├── pipeline/
│   ├── decision_engine.py
│   └── inspection_pipeline.py
│
├── rag/
│   ├── embeddings.py
│   ├── generator.py
│   ├── parser.py
│   ├── retriever.py
│   └── vector_store.py
│
├── rl/
│   ├── dqn_agent.py
│   ├── environment.py
│   ├── pybullet_sim.py
│   ├── robot_env.py
│   ├── train.py
│   └── train_dqn.py
│
├── simulation/
│
├── tests/
│
├── vision/
│   ├── data.yaml
│   ├── detector.py
│   └── train.py
│
├── requirements.txt
└── .gitignore
```


## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app/main.py
```

The application provides the PCB inspection, robotic sorting, and analytics interfaces.

---

## 🎯 Project Outcome

The project provides an end-to-end intelligent PCB inspection workflow:

**Computer Vision → Defect Detection → RAG Analysis → Repair/Reject Decision → Robotic Sorting → Final PCB Report**

It combines **AI-based inspection, knowledge retrieval, decision making, reinforcement learning, and robotic simulation** into a single automated PCB inspection and sorting pipeline.

---

## 👩‍💻 Author

**Saniya Khadapkar**

MSc Artificial Intelligence 

**Interests:**
AI/ML • Computer Vision • Deep Learning • Generative AI • Reinforcement Learning


##screenshort
1.PCB detection page
<img width="1600" height="854" alt="PCB detection page" src="https://github.com/user-attachments/assets/cbaaf879-d473-41b0-ae7f-32fbc56b47c5" />

2.sorting simulation
<img width="1600" height="837" alt="sorting simulation" src="https://github.com/user-attachments/assets/618f1bfe-5de9-4149-ab9b-f2b13338948e" />

3.generated report
<img width="1600" height="831" alt="generated report1" src="https://github.com/user-attachments/assets/057eda41-fe62-4835-910f-980dba574afd" />

<img width="1600" height="841" alt="generated report2" src="https://github.com/user-attachments/assets/aa73d9b7-c66d-47c1-be02-8d12a371bf3c" />



