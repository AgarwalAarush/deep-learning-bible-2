import tensorflow as tf

# Load FFN MNIST Dataset
mnist = tf.keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

# Exploratory Data Analysis
import matplotlib.pyplot as plt
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i], cmap = plt.get_cmap('gray'))
plt.show()

# Build Dataset
batch_size = 100

X_train_new_axis = X_train[..., tf.newaxis]
X_test_new_axis = X_test[..., tf.newaxis]

shuffle_size = 100000

train_ds = tf.data.Dataset.from_tensor_slices((X_train_new_axis, Y_train)).shuffle(shuffle_size).batch(batch_size)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test_new_axis, Y_test)).batch(batch_size)

# Model Engineering
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout 
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np

# Hyperparameters
hidden_size = 256
output_dim = 10
EPOCHS = 30
learning_rate = 1e-3

# Model Architecture
class Feed_Forward_Net(Model):
    def __init__(self):
        super(Feed_Forward_Net, self).__init__()
        self.flatten = Flatten()
        self.layer_1 = Dense(hidden_size, activation = 'relu')
        self.dropout_1 = Dropout(0.2)
        self.layer_2 = Dense(output_dim, activation = 'softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.dropout_1(x)
        output = self.layer_2(x)
        return output

model = Feed_Forward_Net()
model.summary()

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Loss Function
criteria = losses.SparseCategoricalCrossentropy()

train_loss = metrics.Mean(name = 'train_loss')
train_accuracy = metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = metrics.Mean(name = 'test_loss')
test_accuracy = metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        losses = criteria(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradient(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = criteria(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)

# Training Loop
from tqdm import tqdm, notebook, trange

for epoch in range(EPOCHS):
    with notebook.tqdm(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []

        for images, labels in train_ds:

            train_step(images, labels)

            loss_val = train_loss.result()
            acc = train_accuracy.result() * 100

            train_losses.append(loss_val)
            train_accuracies.append(acc)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")

# Model Evaluation
with notebook.tqdm(total_len = len(test_ds), desc = f"Test_ Epoch {epoch + 1}") as pbar:
    test_losses = []
    test_accuracies = []

    for test_images, test_lables in test_ds:
        test_step(test_images, test_lables)

        loss_val = test_loss.result()
        acc = test_accuracy.result() * 100

        test_losses.append(loss_val)
        test_accuracies.append(acc)

        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
