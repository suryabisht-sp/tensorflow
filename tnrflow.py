import os 
# ignoring the unecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# print(tf.__version__)

# initialization tensor

# x=tf.constant(5)
# x=tf.constant([[1,2,3],[4,5,6]])
# x=tf.ones((2,2))
# x=tf.zeros((2,3))
# x=tf.eye(2)
# x=tf.random.uniform((1,2), minval=0, maxval=2)
# x=tf.random.normal((1,2), mean=0, stddev=1)


# x=tf.range(start=0, limit=20, delta=5)
# x=tf.cast(x, dtype=tf.int32)
# print(x)



x=tf.constant([1,2,3])
y=tf.constant([4,5,6])
# z=tf.subtract(y,x)
# z=tf.add(x,y)
# z=tf.divide(x,y)



# z=tf.tensordot(x,y, axes=1)
# z=tf.reduce_sum(x*y, axis=0)
# print(z)






#capturing data using indices
# indices= tf.constant([0,2])
# x_indi=tf.gather(x, indices)
# print(x_indi)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the shapes of the training and test data
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Reshape the data and normalize
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Define the model using the Sequential API
model = keras.Sequential([
    keras.Input(shape=(28 * 28,)),  # Note the comma to make it a tuple
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10)
])

# Print the model summary
print(model.summary())

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

# Evaluate the model
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
