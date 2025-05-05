editor settings
Edit

Mirror cell in tab

delete
Delete cell
Ctrl+M D

more_vert
More cell actions

keyboard_arrow_down
Default title text
[ ] 
Run cell (Ctrl+Enter)
cell has not been executed in this session

executed by nay nay
11:15 PM (1 hour ago)
executed in 0.126s
Show code

Code cell output actions


keyboard_arrow_down
Default title text
[ ] 
Run cell (Ctrl+Enter)
cell has not been executed in this session

executed by nay nay
11:15 PM (1 hour ago)
executed in 1.258s
Show code

Code cell output actions
Epoch 1/3
45/45 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.2670 - loss: 2.1753
Epoch 2/3
45/45 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7473 - loss: 1.5828
Epoch 3/3
45/45 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8600 - loss: 1.0851
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
[ ] 
Run cell (Ctrl+Enter)
cell has not been executed in this session

[ ] 
Run cell (Ctrl+Enter)
cell has not been executed in this session

executed by nay nay
11:15 PM (1 hour ago)
executed in 0.126s
fromfrom google.colab import files
     files.download('my_model.h5')

Code cell output actions


keyboard_arrow_down
Default title text
[ ] 
Run cell (Ctrl+Enter)
cell has not been executed in this session
# @title Default title text
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load dataset
digits = load_digits()
X = digits.images / 16.0
y = to_categorical(digits.target)
X = X.reshape(-1, 8, 8, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential([
    Flatten(input_shape=(8, 8, 1)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
# Corrected indentation for these lines
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3)from google.colab import drive
drive.mount('/content/drive')

# Save model
# Corrected indentation for this line
model.save('my_model.h5')
[ ] 
Run cell (Ctrl+Enter)
cell has not been executed in this session
import matplotlib.pyplot as plt
import numpy as np

# x-axis values
x = np.arange(0, 10, 0.1)

# y-axis values (cosine wave)
y = np.cos(x)

# Plot the cosine wave
plt.plot(x, y)

# Add labels and title
plt.xlabel("x")
plt.ylabel("cos(x)")
plt.title("Cosine Wave")

# Display the plot
plt.show()
