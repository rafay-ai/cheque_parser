FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch (CPU version) to keep the image size small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PaddlePaddle (CPU version) and PaddleOCR
RUN pip install --no-cache-dir paddlepaddle==2.6.2 -i https://mirror.baidu.com/pypi/simple
RUN pip install --no-cache-dir paddleocr==2.9.1

# Install other requirements
RUN pip install --no-cache-dir opencv-python-headless Pillow matplotlib numpy

# Copy the rest of the application code
COPY . .

# Create output directories so they exist natively with correct permissions
RUN mkdir -p output debug_output

# Set the default command. You can override arguments when running.
# Example: docker run -v ${PWD}/images:/app/images cheque-parser --batch images/
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
