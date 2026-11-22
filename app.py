from contextlib import asynccontextmanager
import io
import os
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from mediscan.model import SkinLesionCNN


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MEDISCAN_MODEL_PATH", BASE_DIR / "models" / "skin_lesion_cnn.pt"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

model: SkinLesionCNN | None = None
class_names: list[str] = []
image_size = 224


def load_model() -> None:
    global model, class_names, image_size
    model = None
    class_names = []
    image_size = 224
    if not MODEL_PATH.is_file():
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    class_names = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    model = SkinLesionCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(
    title="MediScan AI",
    description="Research-only skin lesion image classification demo",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ready" if model is not None else "model_unavailable"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model is installed. See the project README.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 25_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions are too large.")
            image = image.convert("RGB").resize((image_size, image_size))
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    tensor = torch.from_numpy(np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    results = [
        {"class": class_names[index], "score": round(float(probabilities[index]), 4)}
        for index in torch.argsort(probabilities, descending=True).tolist()
    ]
    return {
        "predictions": results,
        "notice": "Research demonstration only. This result is not a diagnosis or a substitute for assessment by a qualified clinician.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mediscan.app:app", host="127.0.0.1", port=8000)
