import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, x_val, Y_train, t_val = train_test_split(X_train, Y_train, test_size=0.2)

X_train = (X_train.reshape(-1, 784) / 255).astype(np.float32)
X_test  = (X_test.reshape(-1, 784) / 255).astype(np.float32)
x_val   = (x_val.reshape(-1, 784) / 255).astype(np.float32)
Y_train = np.eye(10)[Y_train].astype(np.float32)
Y_test  = np.eye(10)[Y_test].astype(np.float32)
t_val   = np.eye(10)[t_val].astype(np.float32)

from sklearn.utils import shuffle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

np.random.seed(123)
tf.random.set_seed(123)

# Set Hyperparameters
hidden_size = 200
output_dim = 10
EPOCHS = 30
batch_size = 100
learning_rate = 5e-4

# Build Neural Network
class Feed_Forward_Network(Model):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_size, activation = "sigmoid")
        self.l2 = Dense(hidden_size, activation = 'sigmoid')
        self.l3 = Dense(hidden_size, activation = 'sigmoid')
        self.l4 = Dense(output_dim, activation = 'softmax')

    def call(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        y = self.l4(h3)
        return y

model = Feed_Forward_Network(hidden_size, output_dim)

# Optimizer
# optimizer = optimizers.SGD(learning_rate = learning_rate)
optimizer = optimizers.Adam(learning_rate = learning_rate)

# Define Loss Function
criteria = losses.CategoricalCrossentropy()
train_loss = metrics.Mean()
train_acc = metrics.CategoricalAccuracy()

def compute_loss(t, y):
    return criteria(t, y)

def train_step(x, t):
    with tf.GradientTape() as tape:
        preds = model(x)
        loss = compute_loss(t, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_acc(t, preds)

    return loss

def test_step(x, t):
    preds = model(x)
    loss = compute_loss(t, preds)
    test_loss(loss)
    test_acc(t, preds)
    return loss

# Training
n_batches = X_train.shape[0] // batch_size

for epoch in range(EPOCHS):
    x_, t_ = shuffle(X_train, Y_train)

    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size
        train_step(x_[start:end], t_[start:end])

    print('epoch: {}, loss: {:.3}, acc: {:.3f}'.format(
        epoch + 1,
        train_loss.result(),
        train_acc.result()
    ))

# Evaluation
test_loss = metrics.Mean()
test_acc = metrics.CategoricalAccuracy()

test_step(X_test, Y_test)
print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
    test_loss.result(),
    test_acc.result()
))
