import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import RealTimeSudokuSolver

def show_image(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)
    cv2.waitKey(1)  # Ensure the window updates

# Load and set up Camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use CAP_AVFOUNDATION for macOS
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(3, 1280)  # Set width of the frame
cap.set(4, 720)   # Set height of the frame

# Define model architecture
input_shape = (28, 28, 1)
num_classes = 9

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Load weights from pre-trained model
model.load_weights("digitRecognition.h5")

# Main loop to capture frames and process sudoku solving
old_sudoku = None
while True:
    ret, frame = cap.read()  # Read the frame
    if not ret:
        print("Error: Failed to capture image")
        break

    # Recognize and solve sudoku
    sudoku_frame = RealTimeSudokuSolver.recognize_and_solve_sudoku(frame, model, old_sudoku)
    show_image(sudoku_frame, "Real Time Sudoku Solver", 1066, 600)  # Display the 'solved' image

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop when 'q' is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
