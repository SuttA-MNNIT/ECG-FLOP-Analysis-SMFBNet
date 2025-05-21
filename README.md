# ECG-FLOP-Analysis-SMFBNet
A PyTorch implementation of a multi-branch CNN-GRU architecture for ECG classification with computational profiling using THOP. The model supports both full-lead (12-lead) and reduced-lead (SMFB-Net) configurations to evaluate FLOPs, parameter counts, and efficiency trade-offs.
# SMFB-Net ECG Analysis

This repository contains the implementation of the Selective Multi-Feature Branch Network (SMFB-Net) for ECG classification, focusing on computational efficiency by reducing the number of leads from 12 to an average of 9.8 using a Lead Selection Module (LSM). The code computes Floating Point Operations (FLOPs) and parameters for the baseline 12-lead model and SMFB-Net variants, and generates visualization plots.

## Overview

SMFB-Net is designed for efficient ECG arrhythmia classification, achieving significant computational reductions compared to traditional 12-lead models. The repository includes:
- `compute_flops.py`: Computes FLOPs and parameters for the baseline and SMFB-Net (9 leads).
- `generate_plots.py`: Generates bar plots comparing FLOPs and parameters, saved as PDFs.

### Results
- **Baseline (12 leads)**: FLOPs = 2.566G, Params = 680.457K
- **SMFB-Net (9 leads)**: FLOPs = 1.925G, Params = 510.345K, Reduction = 24.98%
- **SMFB-Net (9.8 leads, interpolated)**: FLOPs = 1.968G, Params = 521.686K, Reduction = 23.31%
- **SMFB-Net (9.8 leads, with LSM)**: FLOPs = 1.771G, Params = 521.686K, Reduction = 30.98%

## Prerequisites

- Python 3.8+
- Google Colab (for Google Drive integration, optional)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SuttA-MNNIT/SMFB-Net-ECG.git
   cd SMFB-Net-ECG
