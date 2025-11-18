from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import date

class UserFilter(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier for the user requesting recommendations",
        examples=["N7BHzi80hxrkDeQBAziZ"]
    )
    
    location: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Filter by specific location in Greater Boston area",
        examples=["Boston", "Cambridge", "Somerville"]
    )
    
    max_rent: Optional[int] = Field(
        None,
        ge=0,
        le=10000,
        description="Maximum monthly rent in USD",
        examples=[2000, 1500]
    )
    
    room_type: Optional[str] = Field(
        None,
        min_length=1,
        max_length=50,
        description="Type of room (e.g., Shared, Private, Studio)",
        examples=["Shared", "Private", "Studio"]
    )
    
    flatmate_gender: Optional[str] = Field(
        None,
        min_length=1,
        max_length=50,
        description="Preferred flatmate gender",
        examples=["Male", "Female", "Mixed", "Any"]
    )
    
    attached_bathroom: Optional[str] = Field(
        None,
        min_length=1,
        max_length=50,
        description="Bathroom availability preference",
        examples=["Yes", "No", "Shared"]
    )
    
    lease_duration_months: Optional[int] = Field(
        None,
        ge=1,
        le=24,
        description="Preferred lease duration in months",
        examples=[6, 12, 18]
    )
    
    available_from: Optional[date] = Field(
        None,
        description="Earliest date the room should be available (ISO 8601 format: YYYY-MM-DD)",
        examples=["2025-01-01", "2025-06-15"]
    )

    limit: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """Validate user_id."""
        if not v or not v.strip():
            raise ValueError("user_id cannot be empty")
        return v.strip()

    @field_validator('location', 'room_type', 'flatmate_gender', 'attached_bathroom')
    @classmethod
    def validate_string_fields(cls, v: Optional[str]) -> Optional[str]:
        """Validate string fields."""
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Filter value cannot be empty")
        return v

    @field_validator('available_from')
    @classmethod
    def validate_available_from(cls, v: Optional[date]) -> Optional[date]:
        """Validate date."""
        if v is not None:
            today = date.today()
            min_date = today - timedelta(days=30)
            max_date = today + timedelta(days=365)
            
            if v < min_date:
                raise ValueError(f"Date too far in past. Min: {min_date.isoformat()}")
            if v > max_date:
                raise ValueError(f"Date too far in future. Max: {max_date.isoformat()}")
        return v

    def has_filters(self) -> bool:
        """Check if any filters are applied."""
        return any([
            self.location,
            self.max_rent is not None,
            self.room_type,
            self.flatmate_gender,
            self.attached_bathroom,
            self.lease_duration_months is not None,
            self.available_from is not None
        ])