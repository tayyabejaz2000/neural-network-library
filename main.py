#import dll_load
import tensorflow as tf

from neuro.nn import activation, layer, losses, models, optimizer

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = tf.transpose(x_train / 255.0, perm=[0, 3, 1, 2])
x_test = tf.transpose(x_test / 255.0, perm=[0, 3, 1, 2])

y_train = tf.one_hot(y_train[..., 0], 10)
y_test = y_test[..., 0]

model = models.Sequential(
    layer.Conv2D(3, 64, (5, 5), padding=2),
    activation.ReLU(),
    layer.Conv2D(64, 32, (3, 3), padding=1, stride=2),
    activation.ReLU(),
    layer.Flatten(),
    layer.Dense(1024, 128),
    activation.ReLU(),
    layer.Dropout(0.1),
    layer.Dense(128, 10),
    activation.StableSoftmax(),
)

loss = losses.CategoricalCrossentropy()
optim = optimizer.Adam()

print("Starting Training")
epochs = 1
batch_size = 128
for i in range(epochs):
    start, end = 0, batch_size
    batch_num = 0
    while end < tf.shape(x_train)[0]:
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        start, end = end, end + batch_size
        # Forward Propagation
        y_pred = model(batch_x)

        # Calculation of Loss
        train_loss = loss(y_pred, batch_y)
        print(
            f"Epoch: {i + 1}, Batch: {batch_num}, Loss: {train_loss.numpy()}")
        batch_num += 1

        # Back Propagation + Optimizing
        optim(model, loss)


model.trainable = False
predictions = model(x_test[:512])

predictions = tf.argmax(predictions, axis=-1)

acc = sum(predictions.numpy() == y_test[:512]) / len(y_test[:512])
print(f"Test Accuracy: {acc * 100}%")
