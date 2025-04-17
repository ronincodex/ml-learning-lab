
# Machine Learning Learning Lab 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

A hands-on repository documenting my machine learning journey, featuring fundamental implementations and Git/GitHub workflow experiences.

![XOR Decision Boundary](docs/xor_decision_boundary.png)

## 📜 Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Git Journey](#-git-journey)
- [Troubleshooting](#-troubleshooting)
- [Future Directions](#-future-directions)
- [Acknowledgements](#-acknowledgements)

## 🚀 Features
1. **XOR Classification** (`xor_classification.py`)  
   - 2-layer neural network solving non-linear separation
   - Visual decision boundary generation

2. **Neural Network from Scratch** (`nn_scratch.py`)  
   - Manual implementation of forward/backward propagation
   - Basic gradient descent implementation

3. **Curriculum Learning** (`curriculum_learning.py`)  
   - Progressive difficulty training strategies
   - Custom difficulty scheduling

4. **3D Linguistic Model** (`linguo-net3D.py`)  
   - Spatial language pattern recognition
   - 3D tensor operations implementation

## 💻 Installation
```bash
git clone git@github.com:ronincodex/ml-learning-lab.git
cd ml-learning-lab
python3 -m venv mlvenv
source mlvenv/bin/activate
pip install -r requirements.txt
```

## 🧪 Usage
### XOR Classification
```bash
python xor_classification.py
# Generates docs/xor_decision_boundary.png
```

### Neural Network from Scratch
```bash
python nn_scratch.py
# Output: Epoch 5000 | Loss: 0.0012 | Accuracy: 100.00%
```

## 🔧 Git Journey
### Initial Setup Challenges
1. **Authentication Issues**  
   ```bash
   # Original error: Permission denied (publickey)
   ssh-keygen -t ed25519 -C "your-email@example.com"
   cat ~/.ssh/id_ed25519.pub # Added to GitHub SSH settings
   ```

2. **Divergent Histories**  
   ```bash
   # Solution for conflicting commits
   git pull origin main --allow-unrelated-histories
   ```

3. **First Successful Push**  
   ```bash
   Enumerating objects: 19, done.
   Writing objects: 100% (18/18), 35.93 KiB | 5.99 MiB/s
   Branch 'main' set up to track remote branch 'main' from 'origin'
   ```

## 🛠️ Troubleshooting
### Common Git Errors
1. **Authentication Failure**
   ```bash
   git remote set-url origin git@github.com:ronincodex/ml-learning-lab.git
   ```

2. **Merge Conflicts**
   ```bash
   git mergetool  # Use vimdiff to resolve conflicts
   git commit -m "Resolved merge conflicts"
   ```

### Python Environment Issues
```bash
# If packages not recognized
deactivate && rm -rf mlvenv && pip install -r requirements.txt
```

## 🌟 Future Directions
- Implement convolutional layers in scratch NN
- Add attention mechanisms to linguo-net3D
- Create interactive Jupyter notebook demonstrations

## 🙏 Acknowledgements
Special thanks to the **GitHub Assistant** who guided me through:
- SSH key configuration challenges
- Git merge conflict resolution strategies
- Repository structure best practices
- Comprehensive documentation practices

Your patient explanations of error messages and workflow optimizations were instrumental in building this learning lab!

---

**Crafted with ❤️ by [RoninCodeX](https://github.com/ronincodex)**  
*"The expert in anything was once a beginner." - Helen Hayes*
