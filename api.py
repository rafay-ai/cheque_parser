from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from main import run_pipeline

app = FastAPI(title="Cheque Parser API")

@app.post("/parse/")
async def parse_cheque(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run the pipeline
        result = run_pipeline(
            image_path=temp_file_path,
            date_model_path="weights/date_model.pt",
            amount_model_path="weights/amount_model.pt",
            use_gpu=False,
            output_dir=None
        )
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up the temporary file in case of an error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Standard entry point for running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
