import numpy as np
import pandas as pd

def sigmoid(z):
    z = np.clip(z, -500, 500)   
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# BatchNorm forward
def batchnorm_forward(Z, gamma, beta, epsilon=1e-8):
    mu = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_hat = (Z - mu) / np.sqrt(var + epsilon)
    out = gamma * Z_hat + beta
    cache = (Z, Z_hat, mu, var, gamma, beta, epsilon)
    return out, cache

# BatchNorm backward
def batchnorm_backward(dout, cache):
    Z, Z_hat, mu, var, gamma, beta, epsilon = cache
    m = Z.shape[0]

    dZ_hat = dout * gamma
    dvar = np.sum(dZ_hat * (Z - mu) * -0.5 * (var + epsilon)**(-1.5), axis=0, keepdims=True)
    dmu = np.sum(dZ_hat * -1/np.sqrt(var + epsilon), axis=0, keepdims=True) + dvar * np.mean(-2*(Z - mu), axis=0, keepdims=True)

    dZ = (dZ_hat / np.sqrt(var + epsilon)) + (dvar * 2 * (Z - mu) / m) + (dmu / m)
    dgamma = np.sum(dout * Z_hat, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    return dZ, dgamma, dbeta


# Load dataset
df = pd.read_csv("Medicaldataset.csv")

testd = df.iloc[1000:,:].copy()
df = df.iloc[:1000,:].copy()

df.iloc[:,-1] = (df.iloc[:,-1] == 'positive').astype(int)
testd.iloc[:,-1] = (testd.iloc[:,-1] == 'positive').astype(int)

Y = df.iloc[:,-1].to_numpy(dtype=int).reshape(-1,1)
X = df.iloc[:,:-1].to_numpy()

# Initialize parameters
np.random.seed(42)  
W1 = np.random.randn(30, X.shape[1]) * 0.01
b1 = np.zeros((1, 30))                 
W2 = np.random.randn(1, 30) * 0.01
b2 = np.zeros((1, 1))                 

# BatchNorm parameters (for hidden layer)
gamma = np.ones((1, 30))
beta = np.zeros((1, 30))

# Hyperparams
learning_rate = 0.001
epochs = 5000
batch_size = 64
beta_mom = 0.9   # momentum factor

v_dW1 = np.zeros_like(W1)
v_db1 = np.zeros_like(b1)
v_dW2 = np.zeros_like(W2)
v_db2 = np.zeros_like(b2)
v_dgamma = np.zeros_like(gamma)
v_dbeta = np.zeros_like(beta)

cost = []

for epoch in range(epochs):
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    epoch_loss = 0
    num_batches = X.shape[0] // batch_size

    for i in range(0, X.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]
        m_batch = X_batch.shape[0]

        # ----- Forward pass -----
        Z1 = np.dot(X_batch, W1.T) + b1         

        # Apply BatchNorm here
        Z1_norm, cache_bn = batchnorm_forward(Z1, gamma, beta)
        
        A1 = sigmoid(Z1_norm)   # activation after BN
        Z2 = np.dot(A1, W2.T) + b2        
        A2 = sigmoid(Z2)

        # Loss
        loss = -(1/m_batch) * np.sum(Y_batch * np.log(A2+1e-8) + (1 - Y_batch) * np.log(1 - A2 + 1e-8))  
        epoch_loss += loss

        # ----- Backward pass -----
        dZ2 = A2 - Y_batch                         
        dW2 = (1/m_batch) * np.dot(dZ2.T, A1)     
        db2 = (1/m_batch) * np.sum(dZ2, axis=0, keepdims=True) 

        dA1 = np.dot(dZ2, W2)               
        dZ1_norm = dA1 * sigmoid_derivative(A1)  

        # Backprop through BatchNorm
        dZ1, dgamma, dbeta = batchnorm_backward(dZ1_norm, cache_bn)

        dW1 = (1/m_batch) * np.dot(dZ1.T, X_batch)     
        db1 = (1/m_batch) * np.sum(dZ1, axis=0, keepdims=True)  

        # ----- Momentum (EWA) update -----
        v_dW1 = beta_mom * v_dW1 + (1 - beta_mom) * dW1
        v_db1 = beta_mom * v_db1 + (1 - beta_mom) * db1
        v_dW2 = beta_mom * v_dW2 + (1 - beta_mom) * dW2
        v_db2 = beta_mom * v_db2 + (1 - beta_mom) * db2
        v_dgamma = beta_mom * v_dgamma + (1 - beta_mom) * dgamma
        v_dbeta = beta_mom * v_dbeta + (1 - beta_mom) * dbeta

        # Update weights
        W1 -= learning_rate * v_dW1
        b1 -= learning_rate * v_db1
        W2 -= learning_rate * v_dW2
        b2 -= learning_rate * v_db2
        gamma -= learning_rate * v_dgamma
        beta -= learning_rate * v_dbeta

    avg_loss = epoch_loss / num_batches
    cost.append(avg_loss)

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
