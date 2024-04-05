
# Data Engineering
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Generate Data
N = 4000
x, t = datasets.make_moons(N, noise=0.3)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,4))
plt.scatter(x[:,0], x[:,1], c=t, cmap=plt.cm.winter)
t = t.reshape(N, 1)

plt.show()

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(x, t, test_size=0.2)

# Import Model Engineering Libraries
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

np.random.seed(123)
tf.random.set_seed(123)

# Set Hyperparameters
hidden_size = 10
output_dim = 1  # output layer dimensionality
EPOCHS = 100
batch_size = 100
learning_rate = 0.1

# Build Neural Network Model
class Feed_Forward_Net(Model):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_size, activation="sigmoid")
        self.l2 = Dense(output_dim, activation="sigmoid")
    
    def call(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y
    
model = Feed_Forward_Net(hidden_size, output_dim)

# Optimizer
optimizer = optimizers.SGD(learning_rate = learning_rate)

# Loss Function
critertion = losses.BinaryCrossentropy()
def compute_loss(t, y):
    return critertion(t, y)

# Training Loop
def train_step(x, t):
    with tf.GradientTape() as tape:
        preds = model(x)
        loss = compute_loss(t, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Testing Loop
def test_step(x, t):
    preds = model(x)
    loss = compute_loss(t, preds)
    test_loss(loss)
    test_acc(t, preds)
    return loss

# Episode/Step Process
n_batches = X_train.shape[0] // batch_size
for epoch in range(EPOCHS):
    train_loss = 0
    x_, t_ = shuffle(X_train, Y_train)

    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size
        loss = train_step(x_[start:end], t_[start:end])
        train_loss += loss.numpy()/n_batches

    print('epoch: {}, loss: {:.3}'.format(epoch + 1, train_loss))

# Model Evaluation
test_loss = metrics.Mean()
test_acc = metrics.BinaryAccuracy()

test_step(X_test, Y_test)

print('test_loss: {:.3f}, test_acc: {:.3f}'.format(test_loss.result(), test_acc.result()))

# # Model Compilation
# model.compile(optimizer=optimizer,
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# 
# model.fit(X_train, Y_train,
#           epochs = EPOCHS, batch_size = batch_size, verbose = 0)
# 
# loss, acc = model.evaluate(X_test, Y_test, verbose = 0)
# print('test_loss: {:.3f}, test_acc: {:.3f}'.format(loss, acc))
