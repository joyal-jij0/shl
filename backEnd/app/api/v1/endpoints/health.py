from fastapi import APIRouter 
from app.schemas import HealthCheckResponse  
from app.services.health_service import get_health_status

router = APIRouter() 

@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    return get_health_status()


