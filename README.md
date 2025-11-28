# SLOPER
SLOPER is a computational framework for modeling spatial transcriptomics data with **inhomogeneous Poisson point processes (IPPPs)** and estimating the spatial gradient of gene expression using **truncated score matching**.


## 🚀 Quick Start

We provide a complete interactive walkthrough in the notebook using the **DLPFC** Visium dataset (#151673):

**See `demo.ipynb`** to learn  
(i) how to train SLOPER to estimate the spatial gradient, and  
(ii) how to run annealed Langevin dynamics to generate enhanced features.

## 📦 Dependencies

To run `demo.ipynb` and the core SLOPER pipeline, you will need the following key libraries (versions used in our examples are listed below):

- **numpy**: 2.1.0  
- **pandas**: 2.2.3  
- **scanpy**: 1.11.1  
- **torch**: 2.5.1+cu124  
- **shapely**: 2.1.1  
- **scikit-learn**: 1.5.2  
- **matplotlib**: 3.10.1  
- **scipy**: 1.15.2  