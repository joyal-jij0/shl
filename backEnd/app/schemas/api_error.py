from typing import Optional, Any
from pydantic import BaseModel
from fastapi.responses import JSONResponse

class ApiError(BaseModel):
    status_code: int 
    data: Optional[Any] = None
    message: str = "Something went wrong"
    success: bool = False 

    def to_response(self):
        return JSONResponse(
            status_code=self.status_code,
            content=self.model_dump()
        )