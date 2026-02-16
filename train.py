from utils import load_mnist
from model import NeuralNetwork 
import numpy as np 
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

X, y = load_mnist()

X = X[:20000]
y = y[:20000]

split = int(0.8*len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
nn = NeuralNetwork(784,256,10,lr=0.03)

epochs = 100

train_accs = []
test_accs = []
losses = []

for epochs in range(epochs):
    y_pred = nn.forward(X_train)
    loss = nn.compute_loss(y_pred, y_train)
    nn.backward(X_train,y_train)

    train_acc = np.mean(nn.predict(X_train)==y_train)
    test_acc = np.mean(nn.predict(X_test) == y_test)

    losses.append(loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)


    print(f"Epoch {epochs+1}, Loss: {loss:4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")  

plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(train_accs, label="Train")
plt.plot(test_accs, label="Test")
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy_curve.png")

plt.show()
