# Phishing Detection Research - Setup Guide

This guide ensures the project is portable and runs with the correct dependencies across different machines.

## 1. Environment Setup

### **Windows**
```powershell
# Create the environment
python -m venv SHAP_based_correction_venv

# Activate the environment
.\SHAP_based_correction_venv\Scripts\activate

# Install dependencies from your file
pip install -r requirements.txt

# --- JUPYTER KERNEL CONFIGURATION ---
# Install the kernel utility
pip install ipykernel

# Register the environment as a Jupyter Kernel
python -m ipykernel install --user --name=SHAP_based_correction_venv --display-name "Python (SHAP_based_correction_venv)"
