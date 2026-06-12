import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img
)

# Load image
img = load_img("Dog.png", target_size=(150, 150))

# Convert image to array
x = img_to_array(img)

# Reshape image
x = np.expand_dims(x, axis=0)

# Create ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True
)

# Generate and display augmented images
i = 0

plt.figure(figsize=(10, 5))

for batch in datagen.flow(x, batch_size=1):

    plt.subplot(1, 4, i + 1)
    plt.imshow(batch[0].astype("uint8"))
    plt.axis("off")

    i += 1

    if i == 4:
        break

plt.suptitle("Augmented Images")
plt.show()