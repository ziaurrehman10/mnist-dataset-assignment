"""
app/routers/model.py
/api/v1/model/info
"""
from fastapi import APIRouter, Depends

from app.schemas.prediction import ModelInfoResponse
from app.services.model_service import ModelService, get_model_service

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/info", response_model=ModelInfoResponse, summary="Model architecture metadata")
def model_info(svc: ModelService = Depends(get_model_service)):
    return svc.info()
