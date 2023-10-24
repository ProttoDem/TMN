import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Підготовка даних
data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Реалізація функції втрат Logistic Loss
def logistic_loss(y_pred, y_true):
    loss = tf.math.log(1 + tf.math.exp(-2 * y_true * y_pred))
    return loss

# Тренування моделі
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

# Тренування моделі з різними функціями втрат
theta, train_losses, test_losses = train(X_train, y_train, X_test, y_test, num_epochs, logistic_loss)

# Візуалізація кривих навчання Logistic Loss
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

# Logistic Loss
y_pred = theta.predict(X_test)
y_pred_binary = (y_pred > threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy \n(Logistic Loss): {accuracy}')

