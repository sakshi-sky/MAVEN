import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from utils import load_iemocap
import os

os.makedirs("outputs", exist_ok=True)

X_train, X_test, y_train, y_test, le, df = load_iemocap()
n_classes = len(le.classes_)

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)
y_train_cat = to_categorical(y_train, n_classes)
y_test_cat  = to_categorical(y_test,  n_classes)

model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_cnn, y_train_cat,
                    epochs=50, batch_size=32,
                    validation_split=0.1, verbose=1)

y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
print(f"CNN Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df['emotion'].value_counts().plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
axes[0].set_title("Emotion Distribution (Dataset)")
axes[0].set_xlabel("Emotion")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)

axes[1].plot(history.history['accuracy'],     label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title("CNN Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[2])
axes[2].set_title("CNN Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/cnn_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/cnn_results.png")