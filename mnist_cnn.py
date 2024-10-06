import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Define the CNN Model
def build_cnn_model_for_marks():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    
    # Output layer for detecting marks from 0 to 100
    model.add(Dense(units=101, activation='softmax'))  # 0 to 100 marks
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the custom data
def load_custom_data_for_marks():
    # Here you would load and preprocess your custom handwritten marks dataset
    # For now, we'll use the MNIST dataset as a placeholder
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 101)  # Placeholder for marks from 0-100
    y_test = to_categorical(y_test, 101)
    return x_train, y_train, x_test, y_test

# Train the Model
def train_cnn_model(model, x_train, y_train, x_test, y_test, epochs=10):
    try:
        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    except Exception as e:
        print(f"Error during training: {e}")

# Save the Model
def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load the custom data
    x_train, y_train, x_test, y_test = load_custom_data_for_marks()

    # Build and train the CNN
    cnn_model = build_cnn_model_for_marks()
    train_cnn_model(cnn_model, x_train, y_train, x_test, y_test, epochs=10)

    # Save the trained model
    save_model(cnn_model, 'handwritten_marks_model.h5')