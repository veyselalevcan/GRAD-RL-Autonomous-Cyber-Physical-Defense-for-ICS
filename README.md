# GRAD-RL: Autonomous Cyber-Physical Defense for ICS

[![Paper](https://img.shields.io/badge/IEEE_Access-DOI-blue)](#) 
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-green)](https://doi.org/10.5281/zenodo.20433521)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview
**GRAD-RL** (Graph-Informed Autonomous Cyber-Physical Defense) is a closed-loop, five-stage autonomous defense pipeline designed for Industrial Control Systems (ICS). It integrates unsupervised anomaly detection, MITRE ATT&CK-aligned threat classification, graph-theoretic risk scoring, Deep Reinforcement Learning (DRL) autonomous response, and an IEC 61511/62443-compliant safety gate.

This repository contains the complete source code, deployment APIs, and environment simulation files used to validate the framework across **GRFICSv3**, **SWaT**, and **WADI** testbeds.

---

## System Architecture & Codebase Structure

The repository is modularized into distinct stages of the cyber-physical defense pipeline:

### 1. Data Processing & Loading (`data_loader.py`)
Handles dynamic dataset switching and strict preprocessing for heterogeneous ICS data.
* **Strict Label Engineering:** Aggregates raw experimental labels into precise MITRE ATT&CK for ICS techniques (e.g., T0836, T0856) to resolve spurious label granularity.
* **Architecture-Agnostic:** Automatically adapts to different sensor dimensionalities (e.g., SWaT 51-dim, WADI 123-dim) without structural code changes.

### 2. Hybrid Detection Pipeline (`train_hybrid.py`)
Trains the core perception and diagnostic layers of the framework.
* **Stage 1 (Perception):** An **LSTM-Autoencoder** trained exclusively on normal operational data to learn the physical invariants of the process. It flags anomalies using a statistically calibrated $P_{99}$ or Transductive Quantile threshold.
* **Stage 2 (Diagnostic):** An **XGBoost Classifier** that takes the anomalous windows and attributes them to specific MITRE ATT&CK techniques, providing SOC-ready intelligence.

### 3. Contextual Risk & Graph Modeling (`topology_manager.py`)
Prevents "topological blindness" by calculating the structural consequence potential of every asset in the ICS network.
* **Graph Centrality:** Models the ICS as a directed graph and calculates Brandes' Betweenness Centrality for all nodes to identify structural bottlenecks.
* **CVSS v4.0 & IEC 61511 Alignment:** Integrates a differentiable risk tensor utilizing a custom $w_s = 1.5$ safety multiplier, prioritizing physical safety impacts over IT-centric metrics.
* **GenericAssetMapper (GAM):** A surjective mapping protocol that translates site-specific tags (e.g., `FIT101`) into a universal 11-stage functional ontology (e.g., `SUPPLY`, `CHEMICAL_DOSING`).

### 4. Reinforcement Learning Environment (`scada_env.py` & `train_rl_agent.py`)
A custom Gymnasium environment built to train the autonomous Proximal Policy Optimization (PPO) Blue Agent.
* **Dual-Mode Observation:** Compresses raw physical telemetry into a 5-dimensional *Abstract Observation Tensor* (Reconstruction Error, Risk Score, Cascade Potential, Operational Health, Dataset Flag) for high-dimensional networks.
* **Reward Shaping:** Balances security mitigation against operational continuity, heavily penalizing unnecessary emergency shutdowns (false positives) and missed critical threats.

### 5. Live Deployment Microservices (`api_detective.py` & `api_defensive.py`)
FastAPI-based microservices designed for real-time (1 Hz Modbus TCP) live deployment.
* **Detective Node:** Ingests streaming telemetry, computes reconstruction errors, predicts MITRE techniques, and generates Culprit Sensor Attribution ($\argmax \varepsilon_j$) for Explainable AI (XAI) reports.
* **Defensive Node:** Receives the intelligence, queries the trained PPO agent, and enforces the **IEC Safety Gate**. It intercepts high-risk recommendations (Risk $\geq$ 9.0 or Emergency Shutdowns) and downgrades them to `REQUIRE_HUMAN_APPROVAL`, ensuring deterministic safety compliance.

---

## Datasets & Environments

The pipeline is validated across three distinct ICS operational domains:
1. **GRFICSv3 (Single-Unit):** 13-feature chemical reactor digital twin. Contains 90,728 samples with explicit MITRE annotations.
2. **SWaT (Sequential):** 51-feature secure water treatment testbed.
3. **WADI (Parallel):** 123-feature water distribution network.

🔗 **The GRFICSv3 MITRE-Annotated Dataset, Live Telemetry, and XAI Logs are publicly available on Zenodo:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20433521.svg)](https://doi.org/10.5281/zenodo.20433521)

---

## Results Summary

| Dataset | LSTM Acc. | FPR | FN | XGB F1 (Macro) |
|---------|-----------|-----|----|--------|
| **GRFICSv3** | 99.70% | 0.44% | 0 | 0.9970 |
| **SWaT** | 99.16% | 1.20% | 0 | 0.8890 |
| **WADI*** | 74.17% | 7.96% | 3,338 | 0.6250 |

*\*WADI FPR represents the performance after applying deployment-specific $P_{92}$ Transductive Quantile Calibration to neutralize extreme inter-dataset domain shifts.*

---

## Installation & Quick Start

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone [https://github.com/veyselalevcan/GRAD-RL-Autonomous-Cyber-Physical-Defense-for-ICS.git](https://github.com/veyselalevcan/GRAD-RL-Autonomous-Cyber-Physical-Defense-for-ICS.git)

# Navigate to the project directory
cd GRAD-RL-Autonomous-Cyber-Physical-Defense-for-ICS

# Install requirements
pip install -r requirements.txt
