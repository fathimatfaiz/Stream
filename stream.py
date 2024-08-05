import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
import tempfile
import time

# Title and Description
st.title("Sentiment Analysis App")
st.write("Building a Machine Learning Application with Streamlit")

# Sidebar for user input
st.sidebar.title("User Input")

# Image file uploader
uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# Video file uploader
uploaded_video = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Display uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to grayscale and flatten
    image = image.convert('L').resize((100, 100))
    image_data = np.array(image).flatten()

    # Create a DataFrame
    df = pd.DataFrame(image_data, columns=['Pixel Value'])
    df['Index'] = df.index

    # Linear Regression Implementation
    X = df[['Index']].values
    y = df['Pixel Value'].values

    # Adding a column of ones for the intercept term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Calculating the optimal weights using the Normal Equation
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # Making predictions
    y_pred = X_b.dot(theta_best)

    # Displaying the results
    st.write("Model Coefficients:", theta_best)
    st.write("Predictions:", y_pred)

    # Plotting the results
    df['Predictions'] = y_pred
    line_chart = alt.Chart(df).mark_line().encode(x='Index', y='Predictions')
    st.altair_chart(line_chart)

# Display uploaded video
if uploaded_video is not None:
    # Save video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Display video
    st.video(tfile.name)

slider_value = st.slider("Select a value", 0, 100)
text_input = st.text_input("Enter some text")

# Progress and status updates with delay
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)

# Generate random data for pie chart
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Value': np.random.randint(1, 150, 5)
})

# Plot pie chart
pie_chart = alt.Chart(data).mark_arc().encode(
    theta=alt.Theta(field="Value", type="quantitative"),
    color=alt.Color(field="Category", type="nominal"),
    tooltip=['Category', 'Value']
)
st.altair_chart(pie_chart)
