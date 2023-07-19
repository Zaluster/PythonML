import tensorflow as tf
import matplotlib.pyplot as plt
from model import MnistModel

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

model=MnistModel()

#ploss and optimizers
loss_function=tf.keras.losses.SparseCategoricalCrossentropy()
optimazer=tf.keras.optimizers.Adam()

train_loss=tf.keras.metrics.Mean(name="train_loss")
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss=tf.keras.metrics.Mean(name="test_loss")
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

@tf.function
def train_step(inputs,outputs):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_function(outputs ,predictions)
    gradients=tape.gradient(loss,model.trainable_variables)
    optimazer.apply_gradients(zip(gradients,model.trainable_variables))

    train_loss(loss)
    train_accuracy(outputs,predictions)

@tf.function
def test_step(inputs, outputs):
    predictions = model(inputs)
    loss = loss_function(outputs, predictions)
    
    test_loss(loss)
    test_accuracy(outputs, predictions)

#format of data
x_train,x_test =x_train/255.0,x_test/255.0
x_train=x_train[...,tf.newaxis]
x_test=x_test[...,tf.newaxis]

train_data= tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)

test_data=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

epochs=5

for epoch in range(epochs):
    for train_inputs,train_labels in train_data:
        train_step(train_inputs,train_labels)

    for test_inputs,test_labels in test_data:
        test_step(test_inputs,test_labels)

    template = 'Epochs: {}, Train loss: {}, Train accuracy: {}, Test loss: {} Test accuracy: {}'
    print(template.format(
        epoch + 1,
        train_loss.result(),
        train_accuracy.result(),
        test_loss.result(),
        test_accuracy.result(),
    ))

train_loss.reset_states()
train_accuracy.reset_states()
test_loss.reset_states()
test_accuracy.reset_states()