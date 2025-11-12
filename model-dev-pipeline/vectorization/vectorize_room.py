"""
Room vectorization module for HomieHub roommate matching.
Converts room listings into normalized 11-dimensional vectors.
"""

import numpy as np
from typing import Dict


def vectorize_room(room_data: Dict, location_coords: Dict[str, tuple]) -> np.ndarray:
    """
    Generate an 11-dimensional vector from room listing.
    
    Vector structure: [lat, lon, gender, rent, lease_duration, room_type,
                      bathroom, food, alcohol, smoke, utilities]
    
    Args:
        room_data: Dictionary containing room details with keys:
            - location: str - Location name (if lat/lon not provided)
            - lat: float - Latitude (optional, overrides location)
            - lon: float - Longitude (optional, overrides location)
            - flatmate_gender: str - "Male", "Female", or "Mixed"
            - rent: int - Monthly rent in dollars
            - lease_duration_months: int - Available lease duration in months
            - room_type: str - "Shared" or "Private"
            - attached_bathroom: str - "Yes" or "No"
            - lifestyle_food: str - "Vegan", "Vegetarian", or "Everything"
            - lifestyle_alcohol: str - "Never", "Rarely", "Occasionally", "Regularly", or "Frequently"
            - lifestyle_smoke: str - "No", "Outside Only", or "Yes"
            - utilities_included: List[str] - List of included utilities
            
        location_coords: Dictionary mapping location names to (lat, lon) tuples
    
    Returns:
        np.ndarray: 11-dimensional normalized vector (float32)
        
    Example:
        >>> location_coords = {"Cambridge": (42.3736, -71.1097)}
        >>> room = {
        ...     "location": "Cambridge",
        ...     "flatmate_gender": "Mixed",
        ...     "rent": 1100,
        ...     "lease_duration_months": 8,
        ...     "room_type": "Shared",
        ...     "attached_bathroom": "No",
        ...     "lifestyle_food": "Vegetarian",
        ...     "lifestyle_alcohol": "Rarely",
        ...     "lifestyle_smoke": "No",
        ...     "utilities_included": ["Heat", "Water", "Gas"]
        ... }
        >>> vector = vectorize_room(room, location_coords)
        >>> vector.shape
        (11,)
    """
    # Normalization constants
    LAT_MIN, LAT_MAX = 42.25, 42.45
    LON_MIN, LON_MAX = -71.20, -71.00
    BUDGET_MIN, BUDGET_MAX = 500, 3000
    LEASE_MIN, LEASE_MAX = 1, 24  # 1 month to 24 months
    
    # Encoding maps
    GENDER_MAP = {"Male": 0.0, "Female": 1.0, "Mixed": 0.5}
    FOOD_MAP = {"Vegan": 0.0, "Vegetarian": 0.5, "Everything": 1.0}
    ALCOHOL_MAP = {
        "Never": 0.0,
        "Rarely": 0.25,
        "Occasionally": 0.5,
        "Regularly": 0.75,
        "Frequently": 1.0
    }
    SMOKE_MAP = {"No": 0.0, "Outside Only": 0.5, "Yes": 1.0}
    
    # 1. Location
    if 'lat' in room_data and 'lon' in room_data:
        lat = room_data['lat']
        lon = room_data['lon']
    else:
        location = room_data.get('location', 'Boston')
        lat, lon = location_coords.get(location, (42.3601, -71.0589))
    
    lat_normalized = max(0.0, min(1.0, (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)))
    lon_normalized = max(0.0, min(1.0, (lon - LON_MIN) / (LON_MAX - LON_MIN)))
    
    # 2. Flatmate gender
    gender = GENDER_MAP.get(room_data.get('flatmate_gender', 'Mixed'), 0.5)
    
    # 3. Rent
    rent = room_data.get('rent', 1500)
    rent_normalized = max(0.0, min(1.0, (rent - BUDGET_MIN) / (BUDGET_MAX - BUDGET_MIN)))
    
    # 4. Lease duration available (in months)
    lease_duration = room_data.get('lease_duration_months', 12)  # Default 12 months
    lease_normalized = max(0.0, min(1.0, (lease_duration - LEASE_MIN) / (LEASE_MAX - LEASE_MIN)))
    
    # 5. Room type
    room_type = 0.0 if room_data.get('room_type', 'Shared') == 'Shared' else 1.0
    
    # 6. Attached bathroom
    bathroom = 0.0 if room_data.get('attached_bathroom', 'No') == 'No' else 1.0
    
    # 7. Food lifestyle
    food = FOOD_MAP.get(room_data.get('lifestyle_food', 'Everything'), 1.0)
    
    # 8. Alcohol lifestyle
    alcohol = ALCOHOL_MAP.get(room_data.get('lifestyle_alcohol', 'Occasionally'), 0.5)
    
    # 9. Smoke lifestyle
    smoke = SMOKE_MAP.get(room_data.get('lifestyle_smoke', 'No'), 0.0)
    
    # 10. Utilities included
    utilities = min(1.0, len(room_data.get('utilities_included', [])) / 4.0)
    
    return np.array([
        lat_normalized, lon_normalized, gender, rent_normalized, lease_normalized,
        room_type, bathroom, food, alcohol, smoke, utilities
    ], dtype=np.float32)