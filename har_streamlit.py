import streamlit as st
import numpy as np
import cv2
import imutils
from collections import deque
import time
import tempfile
import os
import pandas as pd
import altair as alt

# Set page config
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🎥",
    layout="wide"
)

# Title and description
st.title("Real-time Human Activity Recognition")
st.markdown("""
This application uses deep learning to detect and classify human activities in real-time using your webcam or uploaded videos.
The model can recognize hundreds of different activities including sports, daily activities, and more.
""")

# Initialize session state for metrics
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'timestamps': [],
        'fps_values': [],
        'confidence_scores': [],
        'processing_times': [],
        'predictions': [],
        'top_predictions': {}
    }

def reset_metrics():
    """Reset metrics data"""
    st.session_state.metrics = {
        'timestamps': [],
        'fps_values': [],
        'confidence_scores': [],
        'processing_times': [],
        'predictions': [],
        'top_predictions': {}
    }

def update_metrics(fps, confidence_score, processing_time, prediction=None):
    """Update metrics with new values"""
    timestamp = time.time()
    st.session_state.metrics['timestamps'].append(timestamp)
    st.session_state.metrics['fps_values'].append(fps)
    
    # For confidence scores, ensure we're adding a valid value 
    # Even if the model hasn't made a prediction yet, track the confidence
    st.session_state.metrics['confidence_scores'].append(float(confidence_score))
    
    # Ensure processing time is always visible by adding a minimum value
    process_ms = max(processing_time * 1000, 0.1)  # Convert to ms, ensure at least 0.1ms
    st.session_state.metrics['processing_times'].append(process_ms)
    
    if prediction:
        st.session_state.metrics['predictions'].append(prediction)
        
        # Update prediction counts for top activities
        if prediction in st.session_state.metrics['top_predictions']:
            st.session_state.metrics['top_predictions'][prediction] += 1
        else:
            st.session_state.metrics['top_predictions'][prediction] = 1
    
    # Keep only the last 100 points for performance
    max_points = 100
    if len(st.session_state.metrics['timestamps']) > max_points:
        st.session_state.metrics['timestamps'] = st.session_state.metrics['timestamps'][-max_points:]
        st.session_state.metrics['fps_values'] = st.session_state.metrics['fps_values'][-max_points:]
        st.session_state.metrics['confidence_scores'] = st.session_state.metrics['confidence_scores'][-max_points:]
        st.session_state.metrics['processing_times'] = st.session_state.metrics['processing_times'][-max_points:]
        st.session_state.metrics['predictions'] = st.session_state.metrics['predictions'][-max_points:]

def render_performance_metrics():
    """Render performance metrics charts"""
    metrics = st.session_state.metrics
    
    if len(metrics['timestamps']) > 1:
        # Create dataframe for time-series metrics
        df = pd.DataFrame({
            'timestamp': range(len(metrics['timestamps'])),
            'FPS': metrics['fps_values'],
            'Confidence': metrics['confidence_scores'],
            'Processing Time (ms)': metrics['processing_times']
        })
        
        # FPS Chart
        fps_chart = alt.Chart(df).mark_line().encode(
            x=alt.X('timestamp:Q', title='Time'),
            y=alt.Y('FPS:Q', scale=alt.Scale(domain=[0, max(metrics['fps_values']) * 1.2])),
            tooltip=['FPS:Q']
        ).properties(
            title='FPS over time',
            height=150
        ).interactive()
        
        # Confidence Score Chart - Always show full range from 0 to 1
        confidence_chart = alt.Chart(df).mark_line(color='green').encode(
            x=alt.X('timestamp:Q', title='Time'),
            y=alt.Y('Confidence:Q', scale=alt.Scale(domain=[0, 1])),
            tooltip=['Confidence:Q']
        ).properties(
            title='Confidence Score',
            height=150
        ).interactive()
        
        # Processing Time Chart - Set a reasonable minimum scale
        max_time = max(max(metrics['processing_times']), 10)  # At least 10ms scale
        time_chart = alt.Chart(df).mark_line(color='red').encode(
            x=alt.X('timestamp:Q', title='Time'),
            y=alt.Y('Processing Time (ms):Q', scale=alt.Scale(domain=[0, max_time])),
            tooltip=['Processing Time (ms):Q']
        ).properties(
            title='Processing Time (ms)',
            height=150
        ).interactive()
        
        # Top 5 detected activities
        if metrics['top_predictions']:
            top_activities = sorted(metrics['top_predictions'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            activities_df = pd.DataFrame(top_activities, columns=['Activity', 'Count'])
            
            activities_chart = alt.Chart(activities_df).mark_bar().encode(
                y=alt.Y('Activity:N', sort='-x'),
                x=alt.X('Count:Q'),
                color=alt.Color('Activity:N', legend=None)
            ).properties(
                title='Top Detected Activities',
                height=180
            ).interactive()
            
            return fps_chart, confidence_chart, time_chart, activities_chart
        
        return fps_chart, confidence_chart, time_chart, None
    
    return None, None, None, None

def check_cuda_availability():
    """Check if CUDA is available and properly configured with OpenCV"""
    try:
        # Check CUDA availability
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count == 0:
            return False
            
        # Get GPU information
        device = cv2.cuda.getDevice()
        gpu_name = cv2.cuda.printCudaDeviceInfo(device)
        st.sidebar.info(f"📊 GPU Device: NVIDIA GeForce RTX 3060")
        
        # Verify OpenCV CUDA support
        build_info = cv2.getBuildInformation()
        if 'CUDA: YES' not in build_info and 'NVIDIA GPU support: YES' not in build_info:
            return False
            
        return True
    except Exception as e:
        st.sidebar.error(f"CUDA check error: {str(e)}")
        return False

def configure_cuda_model(model):
    """Configure model for CUDA execution"""
    try:
        # Enable CUDA backend
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # Set CUDA target
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # Optional: Enable CUDA optimization
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # Try FP16 for better performance
        return True
    except Exception as e:
        st.sidebar.warning(f"CUDA configuration error: {str(e)}")
        return False

@st.cache_resource
def load_model():
    """Load the HAR model and configure backend based on hardware availability"""
    try:
        model = cv2.dnn.readNet("resnet-34_kinetics.onnx")
        
        # Check CUDA availability and configure
        if check_cuda_availability():
            if configure_cuda_model(model):
                st.sidebar.success("🚀 GPU Acceleration Enabled (RTX 3060)")
            else:
                st.sidebar.warning("⚠️ GPU available but configuration failed. Using CPU.")
                model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            st.sidebar.info("💻 Running on CPU (GPU not available)")
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

@st.cache_data
def load_classes():
    """Load the activity classes"""
    with open("Actions.txt", "r") as f:
        classes = f.read().strip().split("\n")
    return classes

def process_frame(frame, model, classes, frames_deque, confidence_threshold):
    """Process a single frame and return the annotated frame and metrics"""
    start_process_time = time.time()
    frame = imutils.resize(frame, width=400)
    frames_deque.append(frame)
    
    label = None
    max_score = 0.0  # Initialize with 0
    
    if len(frames_deque) == SAMPLE_DURATION:
        # Prepare blob
        blob = cv2.dnn.blobFromImages(frames_deque, 1.0,
                                    (SAMPLE_SIZE, SAMPLE_SIZE),
                                    (114.7748, 107.7354, 99.4750),
                                    swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)
        
        # Get prediction
        model.setInput(blob)
        outputs = model.forward()
        scores = outputs[0]
        max_score = float(np.max(scores))  # Ensure it's a float
        
        if max_score >= confidence_threshold:
            label = classes[np.argmax(scores)]
            # Draw prediction
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{label} ({max_score:.2f})",
                      (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                      0.8, (255, 255, 255), 2)
    
    process_time = time.time() - start_process_time
    return frame, max_score, process_time, label

# Constants
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# Initialize session state for control flow
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Load the model and classes
try:
    model = load_model()
    classes = load_classes()
    st.success("✅ Model and classes loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or classes: {str(e)}")
    st.stop()

# Sidebar controls
st.sidebar.title("Controls")
input_source = st.sidebar.radio("Select Input Source", ["Webcam", "Upload Video"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
show_fps = st.sidebar.checkbox("Show FPS", value=True)

# Create layout for video and metrics
col1, col2 = st.columns([0.6, 0.4])

with col1:
    # Create placeholders for video
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    # Create placeholders for metrics
    st.subheader("Performance Metrics")
    fps_placeholder = st.empty()
    fps_chart_placeholder = st.empty()
    confidence_chart_placeholder = st.empty()
    time_chart_placeholder = st.empty()
    activities_chart_placeholder = st.empty()

if input_source == "Webcam":
    # Webcam setup
    if not st.session_state.processing:
        start_button = st.sidebar.button("Start Webcam Detection", key="start_webcam")
    else:
        stop_button = st.sidebar.button("Stop Webcam Detection", key="stop_webcam")
    
    if not st.session_state.processing and start_button:
        st.session_state.processing = True
        reset_metrics()
        cap = cv2.VideoCapture(0)
        frames = deque(maxlen=SAMPLE_DURATION)
        
        try:
            while st.session_state.processing:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                
                # Process frame
                processed_frame, confidence, process_time, prediction = process_frame(
                    frame, model, classes, frames, confidence_threshold)
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                
                # Update and display metrics
                update_metrics(fps, confidence, process_time, prediction)
                if show_fps:
                    fps_placeholder.text(f"FPS: {fps:.2f}")
                
                # Display frame
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB")
                
                # Update charts
                fps_chart, confidence_chart, time_chart, activities_chart = render_performance_metrics()
                if fps_chart:
                    fps_chart_placeholder.altair_chart(fps_chart, use_container_width=True)
                if confidence_chart:
                    confidence_chart_placeholder.altair_chart(confidence_chart, use_container_width=True)
                if time_chart:
                    time_chart_placeholder.altair_chart(time_chart, use_container_width=True)
                if activities_chart:
                    activities_chart_placeholder.altair_chart(activities_chart, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error during webcam detection: {str(e)}")
        finally:
            cap.release()
            st.session_state.processing = False
            status_placeholder.warning("Webcam released. Click 'Start Detection' to begin again.")

else:
    # Video upload
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        if not st.session_state.processing:
            start_button = st.sidebar.button("Start Video Detection", key="start_video")
        else:
            stop_button = st.sidebar.button("Stop Video Detection", key="stop_video")
            
        if not st.session_state.processing and start_button:
            st.session_state.processing = True
            reset_metrics()
            
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            try:
                # Video processing
                cap = cv2.VideoCapture(tfile.name)
                frames = deque(maxlen=SAMPLE_DURATION)
                
                # Get video info
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Progress bar
                progress_bar = st.progress(0)
                frame_count = 0
                
                while cap.isOpened() and st.session_state.processing:
                    start_time = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame, confidence, process_time, prediction = process_frame(
                        frame, model, classes, frames, confidence_threshold)
                    
                    # Calculate FPS
                    fps = 1.0 / (time.time() - start_time)
                    
                    # Update and display metrics
                    update_metrics(fps, confidence, process_time, prediction)
                    if show_fps:
                        fps_placeholder.text(f"FPS: {fps:.2f}")
                    
                    # Display frame
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB")
                    
                    # Update charts
                    fps_chart, confidence_chart, time_chart, activities_chart = render_performance_metrics()
                    if fps_chart:
                        fps_chart_placeholder.altair_chart(fps_chart, use_container_width=True)
                    if confidence_chart:
                        confidence_chart_placeholder.altair_chart(confidence_chart, use_container_width=True)
                    if time_chart:
                        time_chart_placeholder.altair_chart(time_chart, use_container_width=True)
                    if activities_chart:
                        activities_chart_placeholder.altair_chart(activities_chart, use_container_width=True)
                    
                    # Update progress
                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)
                    
                    # Control playback speed
                    time.sleep(1/video_fps)  # Maintain original video speed
                    
            except Exception as e:
                st.error(f"Error during video processing: {str(e)}")
            finally:
                if 'cap' in locals():
                    cap.release()
                try:
                    # Try to remove temporary file
                    if os.path.exists(tfile.name):
                        os.close(os.open(tfile.name, os.O_RDONLY))  # Close any remaining file handles
                        os.unlink(tfile.name)
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {str(e)}")
                st.session_state.processing = False
                status_placeholder.success("Video processing completed!")

# Stop processing if stop button is clicked
if st.session_state.processing and ('stop_webcam' in locals() and stop_webcam) or ('stop_video' in locals() and stop_video):
    st.session_state.processing = False 