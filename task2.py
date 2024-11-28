import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Simplified CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)


history = model.fit(train_images, train_labels, epochs=15,  # Reduced max epochs
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping], verbose=1)


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

predictions = model.predict(test_images)

# Plot results for a subset
def plot_predictions(images, labels, preds, class_names, num=5):
    plt.figure(figsize=(10, 2 * num))
    for i in range(num):
        plt.subplot(num, 2, 2 * i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[labels[i][0]]}, Pred: {class_names[preds[i].argmax()]}")
        plt.axis('off')
        plt.subplot(num, 2, 2 * i + 2)
        plt.bar(range(10), preds[i], color='gray')
        plt.xticks(range(10), class_names, rotation=45)
        plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

plot_predictions(test_images, test_labels, predictions, class_names)
