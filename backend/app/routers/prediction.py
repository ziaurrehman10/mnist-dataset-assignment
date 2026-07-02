"""
app/routers/prediction.py
/api/v1/predict  and  /api/v1/predict/batch
"""
from fastapi import APIRouter, File, HTTPException, UploadFile, Depends

from app.schemas.prediction import PredictionResponse, BatchPredictionResponse
from app.services.model_service import ModelService, get_model_service
from app.utils.image_utils import image_bytes_to_array

router = APIRouter(prefix="/predict", tags=["Prediction"])

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}
MAX_BATCH_SIZE = 32


def _validate_image(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )


@router.post("", response_model=PredictionResponse, summary="Predict digit from a single image")
async def predict(
    file: UploadFile = File(..., description="Greyscale or colour image of a handwritten digit"),
    svc: ModelService = Depends(get_model_service),
):
    _validate_image(file)
    try:
        raw = await file.read()
        arr = image_bytes_to_array(raw)
        result = svc.predict(arr)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result


@router.post("/batch", response_model=BatchPredictionResponse, summary="Predict digits from multiple images")
async def predict_batch(
    files: list[UploadFile] = File(..., description="Up to 32 images"),
    svc: ModelService = Depends(get_model_service),
):
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {MAX_BATCH_SIZE}. You sent {len(files)}.",
        )
    for f in files:
        _validate_image(f)

    try:
        arrays = [image_bytes_to_array(await f.read()) for f in files]
        predictions = svc.predict_batch(arrays)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"predictions": predictions, "count": len(predictions)}
