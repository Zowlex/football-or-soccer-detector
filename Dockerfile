FROM python:latest

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD detect.py detect.py
ADD export.pkl export.pkl

# Run it once to trigger resnet download
RUN python detect.py

EXPOSE 8008

# Start the server
CMD ["python", "detect.py", "serve"]
