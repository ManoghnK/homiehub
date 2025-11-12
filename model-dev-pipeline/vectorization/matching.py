"""
Matching module for HomieHub roommate matching system.
Implements weighted similarity matching with hard filters for strict criteria.
"""

import numpy as np
from typing import Dict, List, Tuple
from vectorize_user import vectorize_user
from vectorize_room import vectorize_room


# Global weight configuration
# Higher weights = more important in matching
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

# Weight threshold for strict filtering
STRICT_WEIGHT_THRESHOLD = 4.0


def apply_hard_filters(user_data: Dict, rooms_data: List[Dict]) -> List[Dict]:
    """
    Apply hard filters for dimensions with weight >= STRICT_WEIGHT_THRESHOLD.
    Currently filters on:
    - Gender compatibility (weight 4.0)
    - Lease duration compatibility (weight 4.0)
    
    Args:
        user_data: User preferences dictionary
        rooms_data: List of room listing dictionaries
        
    Returns:
        List of rooms that pass hard filters
    """
    filtered_rooms = []
    
    user_gender_pref = user_data.get('gender_preference', 'Any')
    user_lease_months = user_data.get('lease_duration_months', 12)
    
    for room in rooms_data:
        # Hard filter 1: Gender compatibility (weight 4.0)
        room_gender = room.get('flatmate_gender', 'Mixed')
        
        # If user wants specific gender, room must match
        if user_gender_pref == "Male" and room_gender not in ["Male", "Mixed"]:
            continue
        elif user_gender_pref == "Female" and room_gender not in ["Female", "Mixed"]:
            continue
        # If user wants "Any", all rooms are acceptable
        
        # Hard filter 2: Lease duration compatibility (weight 4.0)
        # Room's available lease must be >= user's desired lease
        room_lease_months = room.get('lease_duration_months', 12)
        
        if room_lease_months < user_lease_months:
            # Room lease is shorter than what user needs - exclude
            continue
        
        # Room passed all hard filters
        filtered_rooms.append(room)
    
    return filtered_rooms


def calculate_weighted_distance(
    user_vector: np.ndarray,
    room_vector: np.ndarray,
    weights: np.ndarray = WEIGHTS
) -> float:
    """
    Calculate weighted Euclidean distance between user and room vectors.
    Lower distance = better match.
    
    Formula: sqrt(sum(weight_i² × (user_i - room_i)²))
    
    Args:
        user_vector: User preference vector (11D)
        room_vector: Room listing vector (11D)
        weights: Weight array for each dimension (11D)
        
    Returns:
        float: Weighted Euclidean distance
    """
    weighted_diff = weights * (user_vector - room_vector)
    distance = np.sqrt(np.sum(weighted_diff ** 2))
    return float(distance)


def calculate_weighted_cosine_similarity(
    user_vector: np.ndarray,
    room_vector: np.ndarray,
    weights: np.ndarray = WEIGHTS
) -> float:
    """
    Calculate weighted cosine similarity between user and room vectors.
    Higher similarity = better match.
    
    Args:
        user_vector: User preference vector (11D)
        room_vector: Room listing vector (11D)
        weights: Weight array for each dimension (11D)
        
    Returns:
        float: Weighted cosine similarity (0-1 range)
    """
    # Apply weights to vectors
    weighted_user = weights * user_vector
    weighted_room = weights * room_vector
    
    # Calculate cosine similarity
    dot_product = np.dot(weighted_user, weighted_room)
    magnitude_user = np.linalg.norm(weighted_user)
    magnitude_room = np.linalg.norm(weighted_room)
    
    similarity = dot_product / (magnitude_user * magnitude_room)
    return float(similarity)


def find_best_matches(
    user_data: Dict,
    rooms_data: List[Dict],
    location_coords: Dict[str, Tuple[float, float]],
    top_k: int = 5,
    method: str = 'euclidean',
    apply_filters: bool = True
) -> List[Tuple[Dict, float]]:
    """
    Find top K best room matches for a user using weighted similarity.
    
    Process:
    1. Apply hard filters (if enabled) for strict criteria (weight 4.0)
    2. Generate vectors for user and filtered rooms
    3. Calculate weighted distances
    4. Return top K matches sorted by similarity
    
    Args:
        user_data: User preferences dictionary
        rooms_data: List of room listing dictionaries
        location_coords: Dictionary mapping location names to (lat, lon) tuples
        top_k: Number of top matches to return
        method: 'euclidean' (lower is better) or 'cosine' (higher is better)
        apply_filters: Whether to apply hard filters for strict dimensions
        
    Returns:
        List of (room_dict, score) tuples, sorted by best match first
        
    Example:
        >>> user = {"gender_preference": "Female", "budget_max": 1500, ...}
        >>> rooms = [{"flatmate_gender": "Female", "rent": 1400, ...}, ...]
        >>> location_coords = {"Boston": (42.3601, -71.0589), ...}
        >>> matches = find_best_matches(user, rooms, location_coords, top_k=3)
        >>> for room, score in matches:
        ...     print(f"Room {room['room_id']}: score={score:.4f}")
    """
    # Step 1: Apply hard filters if enabled
    if apply_filters:
        filtered_rooms = apply_hard_filters(user_data, rooms_data)
        if not filtered_rooms:
            # No rooms pass hard filters
            return []
    else:
        filtered_rooms = rooms_data
    
    # Step 2: Generate user vector
    user_vector = vectorize_user(user_data, location_coords)
    
    # Step 3: Calculate similarity scores for all filtered rooms
    matches = []
    for room in filtered_rooms:
        room_vector = vectorize_room(room, location_coords)
        
        if method == 'euclidean':
            score = calculate_weighted_distance(user_vector, room_vector, WEIGHTS)
        elif method == 'cosine':
            score = calculate_weighted_cosine_similarity(user_vector, room_vector, WEIGHTS)
        else:
            raise ValueError("Method must be 'euclidean' or 'cosine'")
        
        matches.append((room, score))
    
    # Step 4: Sort by score
    if method == 'euclidean':
        # Lower distance is better
        matches.sort(key=lambda x: x[1])
    else:  # cosine
        # Higher similarity is better
        matches.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K matches
    return matches[:top_k]


def get_match_explanation(
    user_data: Dict,
    room_data: Dict,
    location_coords: Dict[str, Tuple[float, float]]
) -> Dict:
    """
    Generate detailed explanation of why a room matches (or doesn't match) a user.
    Useful for debugging and showing users why rooms were recommended.
    
    Args:
        user_data: User preferences dictionary
        room_data: Room listing dictionary
        location_coords: Dictionary mapping location names to (lat, lon) tuples
        
    Returns:
        Dictionary with match breakdown by dimension
    """
    user_vector = vectorize_user(user_data, location_coords)
    room_vector = vectorize_room(room_data, location_coords)
    
    dimension_names = [
        'Latitude', 'Longitude', 'Gender', 'Budget', 'Lease Duration',
        'Room Type', 'Bathroom', 'Food', 'Alcohol', 'Smoke', 'Utilities'
    ]
    
    explanation = {
        'overall_distance': calculate_weighted_distance(user_vector, room_vector, WEIGHTS),
        'dimensions': []
    }
    
    for i, name in enumerate(dimension_names):
        diff = abs(user_vector[i] - room_vector[i])
        weighted_diff = WEIGHTS[i] * diff
        
        explanation['dimensions'].append({
            'name': name,
            'weight': float(WEIGHTS[i]),
            'user_value': float(user_vector[i]),
            'room_value': float(room_vector[i]),
            'difference': float(diff),
            'weighted_difference': float(weighted_diff),
            'is_strict': WEIGHTS[i] >= STRICT_WEIGHT_THRESHOLD
        })
    
    return explanation