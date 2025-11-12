"""
User vectorization module for HomieHub roommate matching.
Converts user preferences into weighted 11-dimensional vectors for Firestore.
"""

import numpy as np
from typing import Dict


# Global weight configuration
# Higher weights = more important in matching
# These weights are applied during vectorization for efficient Firestore matching
WEIGHTS = np.array([
    3.0,  # index 0: latitude (location - high priority)
    3.0,  # index 1: longitude (location - high priority)
    4.0,  # index 2: gender (STRICT - highest priority)
    3.0,  # index 3: budget (high priority)
    4.0,  # index 4: lease_duration (STRICT - highest priority)
    2.0,  # index 5: room_type (medium priority)
    1.0,  # index 6: bathroom (low priority)
    1.0,  # index 7: food (low priority)
    1.0,  # index 8: alcohol (low priority)
    1.0,  # index 9: smoke (low priority)
    2.0   # index 10: utilities (medium priority)
], dtype=np.float32)


def vectorize_user(user_data: Dict, location_coords: Dict[str, tuple]) -> np.ndarray:
    """
    Generate a weighted 11-dimensional vector from user preferences.
    
    Weights are applied during vectorization for efficient Firestore similarity matching.
    The resulting weighted vectors can be directly compared using cosine similarity
    or Euclidean distance in Firestore.
    
    Vector structure: [lat, lon, gender, budget, lease_duration, room_type, 
                      bathroom, food, alcohol, smoke, utilities]
    All values are normalized to [0,1] then multiplied by their respective weights.
    
    Args:
        user_data: Dictionary containing user preferences with keys:
            - preferred_locations: List[str] - List of preferred location names
            - gender_preference: str - "Male", "Female", or "Any"
            - budget_max: int - Maximum budget in dollars
            - lease_duration_months: int - Desired lease duration in months
            - room_type_preference: str - "Shared" or "Private"
            - attached_bathroom: str - "Yes" or "No"
            - lifestyle_food: str - "Vegan", "Vegetarian", or "Everything"
            - lifestyle_alcohol: str - "Never", "Rarely", "Occasionally", "Regularly", or "Frequently"
            - lifestyle_smoke: str - "No", "Outside Only", or "Yes"
            - utilities_preference: List[str] - List of desired utilities
            
        location_coords: Dictionary mapping location names to (lat, lon) tuples
    
    Returns:
        np.ndarray: 11-dimensional weighted vector (float32)
        
    Example:
        >>> location_coords = {"Cambridge": (42.3736, -71.1097)}
        >>> user = {
        ...     "preferred_locations": ["Cambridge"],
        ...     "gender_preference": "Any",
        ...     "budget_max": 1200,
        ...     "lease_duration_months": 6,
        ...     "room_type_preference": "Shared",
        ...     "attached_bathroom": "No",
        ...     "lifestyle_food": "Vegetarian",
        ...     "lifestyle_alcohol": "Occasionally",
        ...     "lifestyle_smoke": "No",
        ...     "utilities_preference": ["Heat", "Water"]
        ... }
        >>> vector = vectorize_user(user, location_coords)
        >>> vector.shape
        (11,)
    """
    # Normalization constants
    LAT_MIN, LAT_MAX = 42.25, 42.45
    LON_MIN, LON_MAX = -71.20, -71.00
    BUDGET_MIN, BUDGET_MAX = 500, 3000
    LEASE_MIN, LEASE_MAX = 1, 24  # 1 month to 24 months
    
    # Encoding maps
    GENDER_MAP = {"Male": 0.0, "Female": 1.0, "Any": 0.5}
    FOOD_MAP = {"Vegan": 0.0, "Vegetarian": 0.5, "Everything": 1.0}
    ALCOHOL_MAP = {
        "Never": 0.0,
        "Rarely": 0.25,
        "Occasionally": 0.5,
        "Regularly": 0.75,
        "Frequently": 1.0
    }
    SMOKE_MAP = {"No": 0.0, "Outside Only": 0.5, "Yes": 1.0}
    
    # 1. Location - average of preferred locations
    preferred_locations = user_data.get('preferred_locations', [])
    lats, lons = [], []
    
    for loc in preferred_locations:
        if loc in location_coords:
            lat, lon = location_coords[loc]
            lats.append(lat)
            lons.append(lon)
    
    # Default if no valid locations
    if not lats:
        lats, lons = [42.3601], [-71.0589]
    
    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)
    
    lat_normalized = max(0.0, min(1.0, (avg_lat - LAT_MIN) / (LAT_MAX - LAT_MIN)))
    lon_normalized = max(0.0, min(1.0, (avg_lon - LON_MIN) / (LON_MAX - LON_MIN)))
    
    # 2. Gender preference
    gender = GENDER_MAP.get(user_data.get('gender_preference', 'Any'), 0.5)
    
    # 3. Budget
    budget_max = user_data.get('budget_max', 1500)
    budget_normalized = max(0.0, min(1.0, (budget_max - BUDGET_MIN) / (BUDGET_MAX - BUDGET_MIN)))
    
    # 4. Lease duration preference (in months)
    lease_duration = user_data.get('lease_duration_months', 12)  # Default 12 months
    lease_normalized = max(0.0, min(1.0, (lease_duration - LEASE_MIN) / (LEASE_MAX - LEASE_MIN)))
    
    # 5. Room type
    room_type = 0.0 if user_data.get('room_type_preference', 'Shared') == 'Shared' else 1.0
    
    # 6. Attached bathroom
    bathroom = 0.0 if user_data.get('attached_bathroom', 'No') == 'No' else 1.0
    
    # 7. Food lifestyle
    food = FOOD_MAP.get(user_data.get('lifestyle_food', 'Everything'), 1.0)
    
    # 8. Alcohol lifestyle
    alcohol = ALCOHOL_MAP.get(user_data.get('lifestyle_alcohol', 'Occasionally'), 0.5)
    
    # 9. Smoke lifestyle
    smoke = SMOKE_MAP.get(user_data.get('lifestyle_smoke', 'No'), 0.0)
    
    # 10. Utilities preference
    utilities = min(1.0, len(user_data.get('utilities_preference', [])) / 4.0)
    
    # Create normalized vector (before weighting)
    normalized_vector = np.array([
        lat_normalized, lon_normalized, gender, budget_normalized, lease_normalized,
        room_type, bathroom, food, alcohol, smoke, utilities
    ], dtype=np.float32)
    
    # Apply weights to create final weighted vector
    weighted_vector = normalized_vector * WEIGHTS
    
    return weighted_vector