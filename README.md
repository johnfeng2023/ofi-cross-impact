# OFI Cross-Impact

This repository contains the implementation of several Order Flow Imbalance (OFI) features—Best-Level, Multi-Level, Integrated, and Cross-Asset OFI—based on high-frequency limit order book data.

## Files
- [ofi_feature.py](ofi/ofi_feature.py): Core feature construction functions for Best-Level, Multi-Level, Integrated, and Cross-Asset OFI.
- [main.ipynb](main.ipynb): Demonstration of the full OFI construction pipeline, data simulation, Lasso regression, and visualizations.
- [requirements.txt](requirements.txt): Python package dependencies

## Setup Instructions

1. **Clone the repository**
```
git clone https://github.com/johnfeng2023/ofi-cross-impact.git
cd ofi-cross-impact
```

2. **(Optional but recommended) Create a virtual environment**
```
python3 -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```
3. **Install dependencies**
```
pip install -r requirements.txt
```
4. **Run main.ipynb notebook**
