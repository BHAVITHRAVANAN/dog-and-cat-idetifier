from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

# Define paths to training and validation folders
train_data_dir = "path/to/training/data"  # Replace with your directory path
validation_data_dir = "path/to/validation/data"  # Replace with your directory path

# Set image dimensions
img_width, img_height = 150, 150  # Adjust dimensions as needed

# Data augmentation (optional) - Creates variations of images for better training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Train data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,  # Adjust batch size as needed
    class_mode="binary"
)

# Validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,  # Adjust batch size as needed
    class_mode="binary"
)

# Create the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(
    train_generator,
    epochs=10,  # Adjust epochs as needed
    validation_data=validation_generator
)

# Save the model (optional)
model.save("cat_dog_classifier.h5")
