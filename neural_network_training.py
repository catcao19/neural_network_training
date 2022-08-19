#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
fname = 'neural_network_training.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]


# In[2]:


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize weights & biases.
        Weights should be initialized with values drawn from a normal
        distribution scaled by 0.01.
        Biases are initialized to 0.0.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        #code
        self.biases = np.zeros(n_neurons)
        
    def forward(self, inputs):
        """
        A forward pass through the layer to give z.
        Compute it using np.dot(...) and then add the biases.
        """
        self.inputs = inputs
        #code
        self.z = np.dot(self.inputs, self.weights) + self.biases
        
    def backward(self, dz):
        """
        Backward pass
        """
        # Gradients of weights
        self.dweights = np.dot(self.inputs.T, dz)
        # Gradients of biases
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        # Gradients of inputs
        self.dinputs = np.dot(dz, self.weights.T)


# In[3]:


class ReLu:
    """
    ReLu activation
    """
    def forward(self, z):
        """
        Forward pass
        """
        self.z = z
        #code
        self.activity = np.maximum(0, self.z)

    def backward(self, dactivity):
        """
        Backward pass
        """
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0


# In[4]:


class Softmax:
    
    def forward(self, z):
        """
        """
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs
    
    def backward(self, dprobs):
        """
        """
        # Empty array
        self.dz = np.empty_like(dprobs)
        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            # flatten to a column vector
            prob = prob.reshape(-1, 1)
            # Jacobian matrix
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)


# In[5]:


class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # clip to prevent division by 0
        # clip both sides to not bias up.
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        # negative log likelihoods
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return loss.mean(axis=0)
    
    def backward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # Number of examples in batch and number of classes
        batch_sz, n_class = probs.shape
        # get the gradient
        self.dprobs = -oh_y_true / probs
        # normalize the gradient
        self.dprobs = self.dprobs / batch_sz


# In[6]:


class SGD:
    """
    """
    def __init__(self, learning_rate=1.0):
        # Initialize the optimizer with a learning rate
        self.learning_rate = learning_rate
        
    def update_params(self, layer):
        #code
        layer.weights = layer.weights - self.learning_rate * layer.dweights
        layer.biases = layer.biases - self.learning_rate * layer.dbiases


# In[7]:


def predictions(probs):
    """
    """
    y_preds = np.argmax(probs, axis=1)
    return y_preds


# In[8]:


def accuracy(y_preds, y_true):
    """
    """
    return np.mean(y_preds == y_true)


# In[9]:


#def one_hot_encoding(n_class, y_true):
    #oh_y_true = np.eye(n_class)[y_true]
    #return oh_y_true


# In[10]:


def forward_pass(X, y_true, oh_y_true):
    """
    """
    dense1.forward(X)
    activation1.forward(dense1.z)
    #code
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)
    
    dense3.forward(activation2.activity)
    
    #code
    probs = output_activation.forward(dense3.z)
    loss = crossentropy.forward(probs, oh_y_true)
    return probs, loss


# In[11]:


def backward_pass(probs, y_true, oh_y_true):
    crossentropy.backward(probs, oh_y_true)
    output_activation.backward(crossentropy.dprobs)
    
    dense3.backward(output_activation.dz)
    activation2.backward(dense3.dinputs)
    
    dense2.backward(activation2.dz)
    activation1.backward(dense2.dinputs)
    
    dense1.backward(activation1.dz)


# In[12]:


epochs = 10
n_batch = 25
n_class = 3
dense1 = DenseLayer(3, 4)
activation1 = ReLu()
dense2 = DenseLayer(4, 8)
activation2 = ReLu()
dense3 = DenseLayer(8, 3)
output_activation = Softmax()
crossentropy = CrossEntropyLoss()
optimizer = SGD()


# In[15]:


for epoch in range(epochs):
    print('epoch:', epoch)
    for batch_i in range(n_batch):
        # Get a mini-batch of data from X_train and y_train. It should have batch_sz examples.
        X = np.split(X_train, n_batch)[batch_i]
        y_true = np.split(y_train, n_batch)[batch_i]
        # One-hot encode y_true
        oh_y_true = np.eye(n_class)[y_true]
        # Forward pass
        forward_pass(X, y_true, oh_y_true)
        # Print accuracy and loss
        probs, loss = forward_pass(X, y_true, oh_y_true)
        y_preds = predictions(probs)
        Accuracy = accuracy(y_preds, y_true)
        print('Accuracy is: {}'.format(Accuracy))
        print('Loss is: {}'.format(loss))
        # Backward pass
        backward_pass(probs, y_true, oh_y_true)
        # Update the weights
        optimizer.update_params(dense3)
        optimizer.update_params(dense2)
        optimizer.update_params(dense1)


# In[16]:


test_probs, test_loss = forward_pass(X_test, y_test,np.eye(n_class)[y_test])
test_y_preds = predictions(test_probs)
Accuracy = accuracy(test_y_preds, y_test)
print('Accuracy is: {}'.format(Accuracy))


# In[ ]:




