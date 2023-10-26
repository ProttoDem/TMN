import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def binary_crossentropy_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, "Shape of y_pred and y_true doesn't match"
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    loss = - (y_true * tf.math.log(y_pred + epsilon) / tf.math.log(2.0) + (1 - y_true) * tf.math.log(1 - y_pred + epsilon) / tf.math.log(2.0))
    mean_loss = tf.reduce_mean(loss)

    return mean_loss

def train(X_train,y_train, X_test, y_test, epochs, loss_function):
  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(X_train.shape[1],)),
      tf.keras.layers.Dense(1, activation='sigmoid',use_bias=True)
  ])
  model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
  history=model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test,y_test))
  loss_train=history.history['loss']
  loss_test=history.history['val_loss']
  return model, loss_train, loss_test

num_epochs = 100

theta, train_losses, test_losses  = train(X_train, y_train, X_test, y_test, num_epochs, binary_crossentropy_loss)

plt.plot(train_losses, label='Train')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(test_losses, label='Test')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

threshold = 0.5

y_pred =theta.predict(X_test)
y_pred_binary = (y_pred > threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy \n(Binary crossentropy): {accuracy}')
