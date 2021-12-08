import tensorflow as tf

from neuro.nn import activation, layer, losses, models, optimizer

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model = models.Sequential(
    layer.Flatten(),
    layer.Dense(28 * 28, 512), activation.ReLU(),
    layer.Dropout(0.25),
    layer.Dense(512, 128), activation.ReLU(),
    layer.Dense(128, 16), activation.ReLU(),
    layer.Dense(16, 10), activation.Softmax(),
)

loss = losses.SparseCategoricalCrossentropy()
optim = optimizer.Adam()

epochs = 100
for i in range(epochs):
    # Forward Propagation
    y_pred = model(x_train)

    # Calculation of Loss
    train_loss = loss(y_pred, y_train)
    print(f"Epoch: {i + 1}, Loss: {train_loss.numpy()}")

    # Back Propagation + Optimizing
    optim(model, loss)


model.trainable = False

predictions = model(x_test)
predictions = tf.argmax(predictions, axis=1).numpy()

acc = sum(predictions == y_test) / len(y_test)
print(f"Test Accuracy: {acc * 100}%")
