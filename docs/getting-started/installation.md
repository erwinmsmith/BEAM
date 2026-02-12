# Installation

This guide will help you install BEAM on your system.

---

## System Requirements

BEAM requires the following system specifications:

**Operating System:**

- Linux (Ubuntu 18.04 or later recommended)
- macOS 10.14 or later
- Windows 10/11 with WSL2

**Hardware Requirements:**

| Component | Recommended |
|-----------|-------------|
| Python | 3.9+ |
| PyTorch |  2.0.0+ |
| OpenAI  | 1.0.0+ |

---

## Installation Steps

### Step 1: Clone the Repository

```bash

git clone https://github.com/erwinmsmith/BEAM.git
cd BEAM
pip install -e .
```

### Step 2: Create Virtual Environment

Using conda (recommended):

```bash
conda create -n beam python=3.9
conda activate beam
```

Using venv:

```bash
python -m venv beam-env
source beam-env/bin/activate  # On Linux/macOS
# beam-env\Scripts\activate   # On Windows
```

### Step 3: Install Dependencies

```bash
# LangChain integration
pip install -e ".[langchain]"

# LangGraph integration  
pip install -e ".[langgraph]"

# Bayesian/MCMC support
pip install -e ".[bayesian]"

# All dependencies
pip install -e ".[all]"

# Development tools
pip install -e ".[dev]"
```

---

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Train your first model in 5 minutes
