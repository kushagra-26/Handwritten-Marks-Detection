import cv2
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import fitz  # PyMuPDF for PDF extraction
from tkinter import Tk, filedialog  # For file uploads

# Define the path to the model
model_path = r"C:\Users\kusha\Desktop\java\project\handwritten_marks_model.h5"

# Check if the model exists and load it
if os.path.exists(model_path):
    print("Model file found. Loading the model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Function to extract images from PDFs
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            base_image = doc.extract_image(img[0])
            image_bytes = base_image["image"]
            img_np = np.frombuffer(image_bytes, np.uint8)
            images.append(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE))
    return images

# Preprocess the image
def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)  # Gaussian blur
    thresholded = cv2.adaptiveThreshold(blurred, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10:  # Filter small contours
            digit_regions.append((x, y, w, h))
    
    digit_regions = sorted(digit_regions, key=lambda r: r[0])
    return thresholded, digit_regions

# Predict the digit
def predict_digit(model, digit_image):
    digit_image = cv2.resize(digit_image, (28, 28)).reshape(1, 28, 28, 1)
    digit_image = digit_image.astype('float32') / 255.0
    prediction = model.predict(digit_image)
    predicted_digit = prediction.argmax(axis=-1)[0]
    return predicted_digit

# Save results to Excel
def save_table_to_excel(predicted_digits, excel_path='predicted_marks.xlsx'):
    rows = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    columns = ['a', 'b', 'c', 'd', 'e', 'Total']  # Now, Total will be extracted from the image

    marks_table = np.zeros((5, 6), dtype=int)
    index = 0
    for row in range(5):
        for col in range(6):  # Including the "Total" column, no summing here
            if index < len(predicted_digits):
                marks_table[row, col] = predicted_digits[index]
                index += 1

    df = pd.DataFrame(marks_table, index=rows, columns=columns)
    df.to_excel(excel_path)
    print(f"Results saved to {excel_path}")

# File selection: user can choose an image or a PDF
def select_file():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Images and PDFs", "*.png *.jpg *.jpeg *.pdf")])
    return file_path

# Main function to handle user file upload
def main():
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return
    
    if file_path.lower().endswith(".pdf"):
        images = extract_images_from_pdf(file_path)
        for img_num, image in enumerate(images):
            processed_image, digit_boxes = preprocess_image(image)
            predicted_digits = []
            for (x, y, w, h) in digit_boxes:
                digit_image = processed_image[y:y+h, x:x+w]
                predicted_digit = predict_digit(model, digit_image)
                predicted_digits.append(predicted_digit)
            save_table_to_excel(predicted_digits, excel_path=f'predicted_marks_page_{img_num+1}.xlsx')
    else:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        processed_image, digit_boxes = preprocess_image(image)
        predicted_digits = []
        for (x, y, w, h) in digit_boxes:
            digit_image = processed_image[y:y+h, x:x+w]
            predicted_digit = predict_digit(model, digit_image)
            predicted_digits.append(predicted_digit)
        save_table_to_excel(predicted_digits)

if __name__ == "__main__":
    main()
