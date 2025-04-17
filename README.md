# Adversarial Attacks on Pipelined Ranking Systems

This repository contains the codebase for the research project **Adversarial Attacks on Pipelined Ranking Systems**.

The environment is set up for **Python 3.11**. While CPU execution is supported, **an NVIDIA GPU is highly recommended** to significantly reduce runtime.

---

## Getting Started
Assume running on a Unix-like operating system (Linux or MacOS).
### 1. Set Up a Virtual Environment

Create and activate a virtual environment (assumed to be named `venv`):

```bash
. ./venv/bin/activate
```

### 2. Install Dependencies
If you are using a GPU, ensure PyTorch is installed and compatible with your CUDA version.

Then, install the required Python packages:
```bash
pip install -r requirements.txt
```

## Running Experiments
### Run an Attack and Generate Examples
To extract original documents and generate adversarial examples, uncomment the specified lines in `src/new_attack.py`, then run:

```bash
python -m src.new_attack
```
### Full Evaluation Pipeline
To run the complete evaluation pipeline:

```bash
python -m src.evaluate
```

## Configuration
You can modify experiment configurations and parameters as needed. Please refer to the documentation and inline comments in the `src/` directory for details.
