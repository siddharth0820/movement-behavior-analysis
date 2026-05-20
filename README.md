# QR Tracking Pipeline - Movement Behavior Analysis

This project implements a proof of concept system for analyzing navigation behavior using an image based tracking pipeline and synthetic simulations.

## Overview

The system consists of two main components:

### 1. Image based tracking pipeline
- Detects QR codes using OpenCV
- Estimates a virtual grid structure
- Identifies the missing marker ("hole")
- Reconstructs the path across image frames

### 2. Synthetic simulation
- Generates grid-based movement paths
- Models random and progressively goal-directed strategies
- Evaluates behavior using quantitative metrics

## Files

- `image_tracking_pipeline.py`  
  Processes image data and reconstructs paths from QR code detections

- `synthetic_simulation.py`  
  Generates simulated paths and analyzes navigation strategies

## Metrics

- Step count  
- Path efficiency (based on Manhattan distance)  
- Directional movement distribution  
- Entropy (measure of randomness vs. structure)  

## Requirements

Install dependencies:

```bash
pip install numpy opencv-python matplotlib
