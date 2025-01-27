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
  - Train ResNet models from scratch using **Frankenstein_optimizer**:
    - **ResNet18**: Use the script `run_resnet18.sh` to start training.
    - **ResNet50**: Use the script `run_resnet50.sh` to start training.
  - Both scripts are configured to perform distributed training across **4 GPUs**.
  
- **MAML (Model-Agnostic Meta-Learning)**:
  - A comprehensive implementation of MAML for rapid adaptation to new tasks.
  - To reproduce results, simply run the provided script `main.py`
  
- **Sentiment Analysis**:
  - Employs the BERT model for sentiment analysis based on the IMDB dataset.
  - Two training approaches are provided:
    - **`initial.py`**: Demonstrates fine-tuning starting from the initial weights of the BERT model.
    - **`transfer.py`**: Demonstrates fine-tuning starting from pre-trained BERT model parameters.
---

### 3. `3_Experiment_Tensorflow`
This folder provides experiments implemented in TensorFlow, demonstrating state-of-the-art techniques:

- **Transfer Learning**:
  - Uses EfficientNet for transfer learning on the CIFAR-10 and CIFAR-100 datasets.

- **Simple Overfitting Problem**:
- This experiment addresses a simple overfitting problem and demonstrates the performance of **Frankenstein_optimizer**.
- To reproduce the results, simply run: "python main.py"

---

### 4. `4_Toy_loss_function`
This folder showcases optimization on a variety of toy problems, illustrating the performance of **Frankenstein_optimizer** and other optimizers on common convex and non-convex loss functions:

- **Beale Function**
- **Saddle Point Problem**
- **Rosenbrock Function** (Rose)

#### Features
- Use `example.ipynb` to visualize the trajectories of different optimizers on specified optimization problems.
- Use `main.py` to generate animations including:
  - Each optimizer's learning preferences.
  - The convergence speed on various problems.

---