# Cheque Parser

An automated OCR-based cheque parsing pipeline using PaddleOCR and custom PyTorch CRNN models to robustly extract amounts, dates, payee names, and routing details.

## Dependencies & Setup

All required and tested dependencies are listed in `requirements.txt`.
The working versions of PaddleOCR and PaddlePaddle used in this release are:
- `paddlepaddle==2.6.2`
- `paddleocr==2.9.1`

### Option 1: Using Docker (Recommended)
Running within a Docker container is the easiest way to avoid any system-level library conflicts.

1. Build the Docker image locally:
   ```bash
   docker build -t cheque-parser .
   ```
2. Test a single image and create its JSON parse:
   *(We mount the local `images/` and `output/` folders to the container)*
   ```bash
   docker run --rm -v ${PWD}/images:/app/images -v ${PWD}/output:/app/output cheque-parser --image "images/cheque.jpg" --json --output output/
   ```
3. Run Batch processing on all images in a folder:
   ```bash
   docker run --rm -v ${PWD}/images:/app/images -v ${PWD}/output:/app/output cheque-parser --batch images/
   ```

### Option 2: Local Python Installation

1. Create and activate a Virtual Environment (Optional but highly recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Process a single image:
   ```bash
   python main.py --image "images/cheque.jpg" --json --output output/
   ```
4. Run Batch processing:
   ```bash
   python main.py --batch images/
   ```
   *The system will create sub-directories in `output/` containing the `result.json` files alongside isolated base64 visual crops of the `amount`, `date`, `rupees`, and `pay_to` regions.*

## Project Structure
- `main.py`: The main entry point script for running inference.
- `src/`: Core logic for text extraction and OCR engine abstractions.
- `src/models/`: Structure definition files for custom ML models.
- `weights/`: Trained neural network model checkpoints.
- `images/`: Example test set of images.
- `output/`: Main directory for logging parsed structures and image crop extractions.
- `debug_output/`: Intermediate OCR visualizations used for troubleshooting bounding boxes.
