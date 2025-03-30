from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from utils import train_model_if_needed, predict_price
import asyncio
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

class InputData(BaseModel):
    المساحة: float
    النوع: str
    المدينة: str
    الحي: str = "غير محدد"
    الواجهة: str = "غير محدد"
    الاستخدام: str = "غير محدد"

@app.post("/predict")
async def predict(data: InputData):
    try:
        result = predict_price(data)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    with open("data.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    train_model_if_needed()
    return {"message": "✅ تم رفع الملف وتدريب النموذج بنجاح!"}

async def auto_train_loop():
    while True:
        train_model_if_needed()
        await asyncio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_train_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)