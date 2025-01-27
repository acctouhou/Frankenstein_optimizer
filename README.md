# Frankenstein Optimizer

**Frankenstein_optimizer** is an advanced optimization framework designed to integrate seamlessly with TensorFlow and PyTorch environments. This project demonstrates various experimental applications and showcases optimization techniques across multiple domains.

---

## Project Structure

### 1. `1_PIP`
This folder provides 'pip install' to installing **Frankenstein_optimizer** in both TensorFlow and PyTorch frameworks.

---

### 2. `2_Experiment_Pytorch`
This folder contains experiments implemented in PyTorch, focusing on high-performance training and model optimization:

- **ImageNet Image Classification**:
  - Utilizes the `timm` library for efficient training on multi-GPU setups.
  
- **MAML (Model-Agnostic Meta-Learning)**:
  - A comprehensive implementation of MAML for rapid adaptation to new tasks.
  
- **Sentiment Analysis**:
  - Employs the BERT model for sentiment analysis based on the IMDB dataset.

---

### 3. `3_Experiment_Tensorflow`
This folder provides experiments implemented in TensorFlow, demonstrating state-of-the-art techniques:

- **Transfer Learning**:
  - Uses EfficientNet for transfer learning on the CIFAR-10 and CIFAR-100 datasets.

- **Reproducing Adaptive Gradient Method Results**:
  - Includes a simple test to replicate experiments and analyze the issues raised in *The Marginal Value of Adaptive Gradient Methods in Machine Learning*.

---

### 4. `4_Toy_loss_function`
This folder showcases optimization on a variety of toy problems, illustrating the performance of **Frankenstein_optimizer** on common convex and non-convex loss functions:

- **Beale Function**
- **Saddle Point Problem**
- **Rosenbrock Function** (Rose)
......

These examples provide a clear visualization of the optimizer's behavior in different optimization landscapes.

---