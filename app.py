import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import fitz  # PyMuPDF for PDF extraction
from io import BytesIO

# Define the path to the model
model_path = r"C:\Users\kusha\Desktop\java\project\handwritten_marks_model.h5"

# Check if the model exists and load it
if os.path.exists(model_path):
    st.write("Model file found. Loading the model...")
    model = tf.keras.models.load_model(model_path)
    st.write("Model loaded successfully!")
else:
    st.error(f"Model file not found at {model_path}")
    st.stop()

# Function to extract images from PDFs
def extract_images_from_pdf(pdf_data):
    doc = fitz.open("pdf", pdf_data)
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
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
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
def save_table_to_excel(predicted_digits):
    rows = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    columns = ['a', 'b', 'c', 'd', 'e', 'Total']

    marks_table = np.zeros((5, 6), dtype=int)
    index = 0
    for row in range(5):
        for col in range(5):
            if index < len(predicted_digits):
                marks_table[row, col] = predicted_digits[index]
                index += 1
    marks_table[:, -1] = np.sum(marks_table[:, :-1], axis=1)

    df = pd.DataFrame(marks_table, index=rows, columns=columns)
    buffer = BytesIO()
    df.to_excel(buffer, index=True)
    buffer.seek(0)
    return buffer

# Main Streamlit app
def main():
    st.title("Handwritten Marks Detection and Excel Export")
    
    file = st.file_uploader("Choose an image or PDF file", type=["png", "jpg", "jpeg", "pdf"])
    
    if file is not None:
        if file.name.lower().endswith(".pdf"):
            st.write("Processing PDF file...")
            pdf_data = file.read()
            images = extract_images_from_pdf(pdf_data)
            for img_num, image in enumerate(images):
                processed_image, digit_boxes = preprocess_image(image)
                predicted_digits = []
                for (x, y, w, h) in digit_boxes:
                    digit_image = processed_image[y:y+h, x:x+w]
                    predicted_digit = predict_digit(model, digit_image)
                    predicted_digits.append(predicted_digit)

                excel_buffer = save_table_to_excel(predicted_digits)
                st.download_button(
                    label=f"Download Excel for Page {img_num + 1}",
                    data=excel_buffer,
                    file_name=f'predicted_marks_page_{img_num+1}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        else:
            st.write("Processing image file...")
            image = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            processed_image, digit_boxes = preprocess_image(image)
            predicted_digits = []
            for (x, y, w, h) in digit_boxes:
                digit_image = processed_image[y:y+h, x:x+w]
                predicted_digit = predict_digit(model, digit_image)
                predicted_digits.append(predicted_digit)

            excel_buffer = save_table_to_excel(predicted_digits)
            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name='predicted_marks.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

if __name__ == "__main__":
    main()