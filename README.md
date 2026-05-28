# GRAD-RL: Autonomous Cyber-Physical Defense for ICS

[![Paper](https://img.shields.io/badge/IEEE_Access-DOI-blue)]([PAPER_DOI])
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-green)]([ZENODO_DOI])
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview
GRAD-RL is a five-stage autonomous cyber-physical defense pipeline 
for Industrial Control Systems integrating LSTM-AE anomaly detection, 
MITRE ATT&CK-aligned XGBoost classification, graph-based risk scoring, 
PPO autonomous response, and IEC 61511/62443-compliant safety gate.

## Results Summary
| Dataset | LSTM Acc. | FPR | FN | XGB F1 |
|---------|-----------|-----|----|--------|
| GRFICSv3 | 99.70% | 0.44% | 0 | 0.9970 |
| SWaT | 99.16% | 1.20% | 0 | 0.889 |
| WADI | 74.17% | 24.3% | 3,338 | 0.625 |

## Installation
\`\`\`bash
git clone [https://github.com/veyselalevcan/GRAD-RL-Autonomous-Cyber-Physical-Defense-for-ICS]
cd GRAD-RL
pip install -r requirements.txt
\`\`\`

## Dataset
Available at Zenodo: [DOI_LINK]

## Citation
[BibTeX after acceptance]
