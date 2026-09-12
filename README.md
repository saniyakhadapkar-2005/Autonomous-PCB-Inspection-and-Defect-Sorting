# 🔍 Autonomous PCB Inspection and Defect Sorting

An AI-powered system for **automated PCB defect detection, repairability analysis, and robotic sorting** using Computer Vision, Retrieval-Augmented Generation (RAG), and Reinforcement Learning.

The system analyzes PCB images, identifies manufacturing defects using **YOLO**, retrieves relevant repair knowledge using **RAG**, makes a repair/reject decision, and uses a **Deep Q-Network (DQN)** with **PyBullet simulation** for robotic sorting.



## 🚀 Project Overview

Manual PCB inspection is time-consuming, repetitive, and prone to human error.

This project provides an intelligent automated pipeline that combines:

* 👁️ **Computer Vision** for PCB defect detection
* 🧠 **RAG** for repair knowledge retrieval
* ⚙️ **Decision Engine** for repairability classification
* 🤖 **Deep Reinforcement Learning** for robotic sorting
* 🦾 **PyBullet** for robotic-arm simulation
* 📊 **Streamlit** for an interactive application

### End-to-End Workflow

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
Repairable           Non-Repairable
│                       │
↓                       ↓
Repair Bin            Reject Bin
│                       │
└──────────┬────────────┘
           ↓
   DQN Robotic Agent
           ↓
    PyBullet Simulation
           ↓
    Sorting Action


##🎯 Problem Statement

PCB manufacturing generates defective boards such as:

  0: Damaged Board
  1: Damaged Component
  2: Missing Component
  3: Solder Bridge
  4: Solder Ball

Traditional inspection requires human operators to manually identify defects and decide whether a PCB should be repaired or rejected.

The goal of this project is to build an intelligent system that can:

1. Detect PCB defects automatically.
2. Identify the type of defect.
3. Retrieve relevant repair information.
4. Determine whether the defect is repairable.
5. Decide whether the PCB should go to the repair or reject bin.
6. Simulate robotic sorting using Reinforcement Learning.



# 🧰 Technology Stack

| Area                   | Technology                      |
| ---------------------- | ------------------------------- |
| Programming            | Python                          |
| Computer Vision        | OpenCV                          |
| Object Detection       | YOLO                            |
| Deep Learning          | PyTorch                         |
| NLP / RAG              | Embeddings + Vector Retrieval   |
| Knowledge Base         | Text-based PCB repair knowledge |
| Reinforcement Learning | DQN                             |
| Robotics Simulation    | PyBullet                        |
| Environment            | Gymnasium-style RL environment  |
| Web UI                 | Streamlit                       |
| Configuration          | YAML                            |
| Version Control        | Git & GitHub                    |

-
