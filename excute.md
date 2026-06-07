use python 3.8.10

download link : https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe

1. pip install -r requirements.txt

2. streamlit run har_streamlit.py (This is Streamlit Application for video upload detection and live webcam detection)

   ```Note : **if you use first webcam then refresh before switching to the upload video
          **if you use first upload video then refresh before switching to the webcam```

3. py HAR.py --model resnet-34_kinetics.onnx --classes Actions.txt (Only for live webcam)
