import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from ultralytics import YOLO

def predict_image(img, model):
    result = model.predict(img)[0].plot()
    return result

def main():
    with st.sidebar:
        st.title("About")
        st.markdown(
            "- **Detection of Military Aircraft using Object Detection**.\n"
            "- **Upload images** to identify aircraft."
        )

    st.title("Military Aircraft Detector")
    
    model_path = r"C:\Users\sivab\.cache\kagglehub\datasets\a2015003713\militaryaircraftdetectiondataset\versions\86\yolo_model_best.pt"
    model = YOLO(model_path)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "gif"])
    submit_button = st.button("Submit")

    if submit_button:
        if not uploaded_file:
            st.error("Please upload a file first!")
        else:
            img = plt.imread(uploaded_file)
            with st.spinner('Detecting aircraft...'):
                pred = predict_image(img, model)
            st.success("Detection complete!")
            st.write("Detected Image:")
            st.image(pred, width=800, channels="RGB")

if __name__ == "__main__":
    main()
