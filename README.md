# 🔍 Autonomous PCB Inspection and Defect Sorting

An AI-powered system for **automated PCB defect detection, repairability analysis, and robotic sorting** using Computer Vision, Retrieval-Augmented Generation (RAG), and Reinforcement Learning.

The system analyzes PCB images, identifies manufacturing defects using **YOLO**, retrieves relevant repair knowledge using **RAG**, makes a repair/reject decision, and uses a **Deep Q-Network (DQN)** with **PyBullet simulation** for robotic sorting.

---

## 🚀 Project Overview

Manual PCB inspection is time-consuming, repetitive, and prone to human error.

This project provides an intelligent automated pipeline that combines:

* 👁️ **Computer Vision** for PCB defect detection
* 🧠 **RAG** for repair knowledge retrieval
* ⚙️ **Decision Engine** for repairability classification
* 🤖 **Deep Reinforcement Learning** for robotic sorting
* 🦾 **PyBullet** for robotic-arm simulation
* 📊 **Streamlit** for an interactive application

### 🔄 End-to-End Workflow

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

---

## ✨ Key Features

* **Automated PCB Defect Detection** using YOLO
* **Defect Classification** with bounding-box detection
* **RAG-based Knowledge Retrieval** for repair information
* **Repair / Reject Decision Engine**
* **DQN-based Robotic Sorting**
* **PyBullet Robotic Simulation**
* **Interactive Streamlit Dashboard**
* Modular Python project architecture

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
| Environment            | Gymnasium                     |
| Web UI                 | Streamlit                     |
| Configuration          | YAML                          |
| Version Control        | Git & GitHub                  |

---

## 📁 Project Structure

```text
Autonomous-PCB-Inspection-and-Defect-Sorting/
│
├── app/             → Streamlit application
├── config/          → Configuration files
├── knowledge_base/  → PCB repair knowledge
├── pipeline/        → Inspection & decision pipeline
├── rag/             → RAG components
├── rl/              → DQN & robotic environment
├── simulation/      → Simulation components
├── tests/           → Testing utilities
├── vision/          → YOLO detection & training
│
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/saniyakhadapkar-2005/Autonomous-PCB-Inspection-and-Defect-Sorting.git
cd Autonomous-PCB-Inspection-and-Defect-Sorting
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app/main.py
```



## 🎯 Project Outcome

The project demonstrates an end-to-end AI pipeline combining:

**Computer Vision → RAG → Decision Making → Reinforcement Learning → Robotic Simulation**

for intelligent PCB inspection and automated defect-based sorting.

---

## 👩‍💻 Author

**Saniya Khadapkar**

MSc Artificial Intelligence Student

**Interests:** AI/ML • Computer Vision • Deep Learning • Generative AI • Reinforcement Learning
