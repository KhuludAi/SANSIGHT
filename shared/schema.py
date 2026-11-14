from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator

from shared.config import VITALS


class Reading(BaseModel):
    astronaut_id: str
    timestamp: datetime
    vital: str
    value: float
    unit: Optional[str] = None
    source: Optional[str] = "sensor"

    @field_validator("vital")
    @classmethod
    def vital_must_exist(cls, v):
        if v not in VITALS:
            raise ValueError(f"Unknown vital '{v}'. Allowed: {list(VITALS)}")
        return v


class Status(BaseModel):
    astronaut_id: str
    vital: str
    value: float
    timestamp: datetime
    state: str  # "normal", "low" or "high"