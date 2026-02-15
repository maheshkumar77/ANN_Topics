import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

df=pd.read_csv('/content/diabetes_prediction_dataset.csv')
df.head()
df['gender']=(df['gender']=='Male').astype(int)
df['smoking_history'] = df['smoking_history'].map({
    'never': 0,
    'former': 1,
    'current': 2,
    'not current': 3
}).fillna(-1)
X = df.drop("diabetes", axis=1)   # Input features
y = df["diabetes"]                # Target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Training Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=20, batch_size=32)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
print("Test loss:", loss)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print('*******************************')
print(classification_report(y_test, y_pred))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Test"])
plt.show()
plt.plot(history.history['loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train"])
plt.show()