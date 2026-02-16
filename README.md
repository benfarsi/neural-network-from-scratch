This project implements a fully connected neural network from scratch using only NumPy.  

No deep learning frameworks (PyTorch, TensorFlow, etc.) were used.



The goal was to implement forward propagation, backpropagation, and gradient descent manually to build a strong understanding of neural network fundamentals.



The model is trained on the MNIST handwritten digit classification dataset.



#### **Architecture**



Input Layer: 784 neurons (28x28 flattened image)  

Hidden Layer: 256 neurons, ReLU activation  

Output Layer: 10 neurons, Softmax activation  



784 → 256 → 10



#### **Forward Pass**



Z1 = XW1 + b1  

A1 = ReLU(Z1)  

Z2 = A1W2 + b2  

A2 = Softmax(Z2)



Softmax is implemented with numerical stability adjustment.



Cross-Entropy Loss:



L = -Σylog(ŷ)



#### **Backpropagation**

Output layer gradient:



dZ2 = A2 - y  



Hidden layer gradient:



dZ1 = (dZ2 W2ᵀ) ⊙ ReLU'(Z1)



Weight updates performed using full-batch gradient descent.



#### **Training Details**

Dataset size: 20,000 samples  

Train/Test split: 80/20  

Learning rate: 0.3  

Epochs: 100  

Optimizer: Full-batch Gradient Descent  



#### **Results**

Final Train Accuracy: ~80%  

Final Test Accuracy: ~79–80%



Training Loss Curve

!\[Loss Curve](loss\_curve.png)



Accuracy Curve

!\[Accuracy Curve](accuracy\_curve.png)



#### **Key Takeaways**

\- Implemented backpropagation manually.

\- Verified gradient flow and convergence behavior.

\- Explored learning rate impact on convergence.

\- Observed generalization gap between train and test sets.



#### **Requirements**

numpy  

scikit-learn  

matplotlib  



dependencies**:**

pip install -r requirements.txt



#### **Future Improvements**



\- Mini-batch gradient descent

\- L2 regularization

\- Adam optimizer

\- Additional hidden layers

\- Convolutional architecture

