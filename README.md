# 🧠 Medical Diagnosis Neural Network (From Scratch in NumPy)

This project implements a **two-layer neural network** (built completely from scratch using NumPy) to predict **medical test results** — whether a patient is *positive* or *negative* for a certain condition — based on input features.

The goal is to demonstrate how **neural networks** work internally, including **forward propagation**, **backpropagation**, **cost calculation**, and **parameter updates**, all without using any deep learning libraries like TensorFlow or PyTorch.

---

## 🚀 Features

- Custom **sigmoid activation** and **sigmoid derivative** functions  
- **Fully vectorized** forward and backward propagation using NumPy  
- **Binary cross-entropy** loss function for classification  
- Adjustable **hidden layer size**, **learning rate**, and **epochs**  
- Tracks and visualizes **training loss** over epochs  
- Includes a simple **accuracy testing** function on unseen data  

---

## 🧩 Network Architecture

| Layer | Type | Size | Activation |
|:------|:------|:------|:------------|
| Input Layer | — | `n_features` | — |
| Hidden Layer | Dense | 30 neurons | Sigmoid |
| Output Layer | Dense | 1 neuron | Sigmoid |

---

## ⚙️ Workflow

1. **Import Data**
   - Loads `Medicaldataset.csv`
   - Splits into:
     - Training data → first 1000 samples  
     - Test data → remaining samples  
   - Converts the target column:  
     `"positive"` → `1`  
     `"negative"` → `0`

2. **Initialize Parameters**
   - Random weights (`W1`, `W2`) using `np.random.randn()`
   - Zero biases (`b1`, `b2`)

3. **Forward Propagation**
   - Compute `Z1 = X·W1ᵀ + b1`
   - Apply sigmoid → `A1`
   - Compute `Z2 = A1·W2ᵀ + b2`
   - Apply sigmoid → `A2`
   - Compute **binary cross-entropy loss**

4. **Backward Propagation**
   - Derive gradients: `dW1`, `db1`, `dW2`, `db2`
   - Update weights with **gradient descent**

5. **Training**
   - Run for up to `10,000,000` epochs (can be reduced)
   - Print loss every `1000` epochs
   - Track cost history for plotting

6. **Testing**
   - Each test sample is passed through the trained model
   - Compare predictions vs. true labels
   - Display overall **accuracy**

---

## 📈 Visualization

The loss curve is plotted using **Seaborn**:

```python
sns.lineplot(cost)
