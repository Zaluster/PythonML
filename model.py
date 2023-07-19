from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import Model

class MnistModel(Model):
    def __init__(self) -> None:
        super(MnistModel,self).__init__()
        self.conv1=Conv2D(32,3,activation='relu')
        self.flatten=Flatten()
        self.dense1=Dense(128,activation='relu')
        self.dense2=Dense(10,activation='softmax')

    def call(self,x):
        x1=self.conv1(x)
        x2=self.flatten(x1)
        x3=self.dense1(x2)
        return self.dense2(x3)
        