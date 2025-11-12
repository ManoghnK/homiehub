"""
Main script for testing HomieHub vectorization and matching system.
Demonstrates end-to-end usage of user/room vectorization and weighted matching.
"""

from typing import Dict, List
from vectorize_user import vectorize_user
from vectorize_room import vectorize_room
from matching import (
    find_best_matches,
    get_match_explanation,
    apply_hard_filters,
    WEIGHTS,
    STRICT_WEIGHT_THRESHOLD
)


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def main():
    """Main test function demonstrating the matching system."""
    
    # Location coordinates mapping
    location_coords = {
        "Cambridge": (42.3736, -71.1097),
        "Boston": (42.3601, -71.0589),
        "Somerville": (42.3876, -71.0995),
        "Back Bay": (42.3505, -71.0763),
        "South End": (42.3414, -71.0742),
        "Fenway": (42.3467, -71.0972),
        "Allston": (42.3543, -71.1312),
    }
    
    # Sample users
    users = [
        {
            "user_id": "U001",
            "name": "Alex",
            "preferred_locations": ["Cambridge", "Somerville"],
            "gender": "Male",
            "gender_preference": "Any",
            "budget_max": 1200,
            "lease_duration_months": 6,
            "room_type_preference": "Shared",
            "attached_bathroom": "No",
            "lifestyle_food": "Vegetarian",
            "lifestyle_alcohol": "Occasionally",
            "lifestyle_smoke": "No",
            "utilities_preference": ["Heat", "Water"]
        },
        {
            "user_id": "U002",
            "name": "Maria",
            "preferred_locations": ["Boston", "Back Bay"],
            "gender": "Female",
            "gender_preference": "Female",
            "budget_max": 1800,
            "lease_duration_months": 12,
            "room_type_preference": "Private",
            "attached_bathroom": "Yes",
            "lifestyle_food": "Everything",
            "lifestyle_alcohol": "Regularly",
            "lifestyle_smoke": "No",
            "utilities_preference": ["Heat", "Water", "Electricity"]
        }
    ]
    
    # Sample rooms
    rooms = [
        {
            "room_id": "R001",
            "location": "Cambridge",
            "flatmate_gender": "Mixed",
            "rent": 1100,
            "lease_duration_months": 8,
            "room_type": "Shared",
            "attached_bathroom": "No",
            "lifestyle_food": "Vegetarian",
            "lifestyle_alcohol": "Rarely",
            "lifestyle_smoke": "No",
            "utilities_included": ["Heat", "Water", "Gas"]
        },
        {
            "room_id": "R002",
            "location": "Boston",
            "flatmate_gender": "Female",
            "rent": 1750,
            "lease_duration_months": 12,
            "room_type": "Private",
            "attached_bathroom": "Yes",
            "lifestyle_food": "Everything",
            "lifestyle_alcohol": "Regularly",
            "lifestyle_smoke": "No",
            "utilities_included": ["Heat", "Water", "Electricity", "Gas"]
        },
        {
            "room_id": "R003",
            "location": "Somerville",
            "flatmate_gender": "Male",
            "rent": 950,
            "lease_duration_months": 3,  # Too short for most users
            "room_type": "Shared",
            "attached_bathroom": "No",
            "lifestyle_food": "Everything",
            "lifestyle_alcohol": "Frequently",
            "lifestyle_smoke": "Outside Only",
            "utilities_included": ["Heat"]
        },
        {
            "room_id": "R004",
            "location": "Back Bay",
            "flatmate_gender": "Female",
            "rent": 1600,
            "lease_duration_months": 10,
            "room_type": "Private",
            "attached_bathroom": "No",
            "lifestyle_food": "Vegetarian",
            "lifestyle_alcohol": "Occasionally",
            "lifestyle_smoke": "No",
            "utilities_included": ["Heat", "Water"]
        },
        {
            "room_id": "R005",
            "location": "Allston",
            "flatmate_gender": "Male",
            "rent": 1050,
            "lease_duration_months": 6,
            "room_type": "Shared",
            "attached_bathroom": "No",
            "lifestyle_food": "Everything",
            "lifestyle_alcohol": "Frequently",
            "lifestyle_smoke": "No",
            "utilities_included": ["Heat", "Water"]
        }
    ]
    
    print_section("HOMIE HUB - WEIGHTED MATCHING SYSTEM TEST")
    
    # Display weight configuration
    print_section("WEIGHT CONFIGURATION", "-")
    dimension_names = [
        'Latitude', 'Longitude', 'Gender', 'Budget', 'Lease Duration',
        'Room Type', 'Bathroom', 'Food', 'Alcohol', 'Smoke', 'Utilities'
    ]
    
    print(f"\nStrict filter threshold: {STRICT_WEIGHT_THRESHOLD}")
    print(f"Dimensions with weight >= {STRICT_WEIGHT_THRESHOLD} will be strictly enforced.\n")
    
    print(f"{'Index':<6} {'Dimension':<15} {'Weight':<8} {'Priority':<15} {'Enforcement'}")
    print("-" * 70)
    for i, (name, weight) in enumerate(zip(dimension_names, WEIGHTS)):
        priority = "STRICT" if weight >= 4.0 else "HIGH" if weight >= 3.0 else "MEDIUM" if weight >= 2.0 else "LOW"
        enforcement = "STRICT ✓" if weight >= STRICT_WEIGHT_THRESHOLD else "WEIGHTED"
        print(f"{i:<6} {name:<15} {weight:<8.1f} {priority:<15} {enforcement}")
    
    # Display users
    print_section("USER PROFILES", "-")
    for user in users:
        print(f"\n{user['name']} ({user['user_id']})")
        print(f"  Looking for: {user['room_type_preference']} room")
        print(f"  Budget: ${user['budget_max']}/month")
        print(f"  Lease: {user['lease_duration_months']} months")
        print(f"  Gender preference: {user['gender_preference']}")
        print(f"  Locations: {', '.join(user['preferred_locations'])}")
    
    # Display rooms
    print_section("AVAILABLE ROOMS", "-")
    for room in rooms:
        print(f"\n{room['room_id']} - {room['location']}")
        print(f"  Type: {room['room_type']}, Flatmates: {room['flatmate_gender']}")
        print(f"  Rent: ${room['rent']}/month")
        print(f"  Lease: {room['lease_duration_months']} months")
        print(f"  Utilities: {', '.join(room['utilities_included'])}")
    
    # Find matches for each user
    print_section("MATCHING RESULTS")
    
    for user in users:
        print(f"\n{'=' * 80}")
        print(f"{user['name']}'s Matches")
        print('=' * 80)
        
        # Get perfect matches (with hard filters)
        perfect_matches = find_best_matches(
            user_data=user,
            rooms_data=rooms,
            location_coords=location_coords,
            top_k=10,  # Get all possible matches
            method='euclidean',
            apply_filters=True
        )
        
        # Get all matches (without hard filters)
        all_matches = find_best_matches(
            user_data=user,
            rooms_data=rooms,
            location_coords=location_coords,
            top_k=10,  # Get all possible matches
            method='euclidean',
            apply_filters=False
        )
        
        # Separate into perfect matches and other close matches
        perfect_room_ids = {room['room_id'] for room, _ in perfect_matches}
        other_matches = [(room, score) for room, score in all_matches 
                        if room['room_id'] not in perfect_room_ids]
        
        # Display Perfect Matches
        print("\n  ✨ PERFECT MATCHES (Meet all strict criteria)")
        print("  " + "-" * 76)
        
        if perfect_matches:
            print(f"  Found {len(perfect_matches)} room(s) matching gender & lease requirements\n")
            
            for rank, (room, distance) in enumerate(perfect_matches, 1):
                print(f"  {rank}. {room['room_id']} - {room['location']}")
                print(f"     Match Score: {distance:.4f} ⭐ (lower is better)")
                print(f"     ${room['rent']}/mo | {room['lease_duration_months']} month lease | "
                      f"{room['room_type']} | {room['flatmate_gender']} flatmates")
                print(f"     Bathroom: {room['attached_bathroom']} | "
                      f"Utilities: {', '.join(room['utilities_included'][:2])}{'...' if len(room['utilities_included']) > 2 else ''}")
                print()
        else:
            print(f"  No rooms meet strict criteria:")
            print(f"  • Gender preference: {user['gender_preference']}")
            print(f"  • Minimum lease: {user['lease_duration_months']} months\n")
        
        # Display Other Close Matches
        if other_matches:
            print("\n  💡 OTHER CLOSE MATCHES (Similar preferences, but don't meet all strict criteria)")
            print("  " + "-" * 76)
            print(f"  Found {len(other_matches)} additional room(s) worth considering\n")
            
            for rank, (room, distance) in enumerate(other_matches, 1):
                # Identify why it didn't pass strict filters
                reasons = []
                
                # Check gender mismatch
                user_gender_pref = user['gender_preference']
                room_gender = room['flatmate_gender']
                if user_gender_pref == "Male" and room_gender not in ["Male", "Mixed"]:
                    reasons.append(f"Gender: {room_gender}")
                elif user_gender_pref == "Female" and room_gender not in ["Female", "Mixed"]:
                    reasons.append(f"Gender: {room_gender}")
                
                # Check lease duration
                if room['lease_duration_months'] < user['lease_duration_months']:
                    reasons.append(f"Lease: {room['lease_duration_months']}mo < {user['lease_duration_months']}mo needed")
                
                reason_str = " | ".join(reasons) if reasons else "See details"
                
                print(f"  {rank}. {room['room_id']} - {room['location']}")
                print(f"     Match Score: {distance:.4f} (lower is better)")
                print(f"     ${room['rent']}/mo | {room['lease_duration_months']} month lease | "
                      f"{room['room_type']} | {room['flatmate_gender']} flatmates")
                print(f"     ⚠️  Why not perfect: {reason_str}")
                print()
    
    # Detailed match explanation for first user's top match
    print_section("DETAILED MATCH EXPLANATION")
    
    user = users[0]
    matches = find_best_matches(user, rooms, location_coords, top_k=1, apply_filters=True)
    
    if matches:
        room, distance = matches[0]
        print(f"\nWhy {room['room_id']} is the best match for {user['name']}:")
        print(f"Overall weighted distance: {distance:.4f}\n")
        
        explanation = get_match_explanation(user, room, location_coords)
        
        print(f"{'Dimension':<15} {'Weight':<8} {'User':<8} {'Room':<8} {'Diff':<8} {'Weighted':<10} {'Status'}")
        print("-" * 80)
        
        for dim in explanation['dimensions']:
            status = "✓ MATCH" if dim['difference'] < 0.1 else "≈ CLOSE" if dim['difference'] < 0.3 else "✗ DIFF"
            strict = " (STRICT)" if dim['is_strict'] else ""
            
            print(f"{dim['name']:<15} {dim['weight']:<8.1f} {dim['user_value']:<8.3f} "
                  f"{dim['room_value']:<8.3f} {dim['difference']:<8.3f} "
                  f"{dim['weighted_difference']:<10.3f} {status}{strict}")
    
    print_section("TEST COMPLETED SUCCESSFULLY")
    print("\nKey Features:")
    print("  ✨ Perfect Matches: Rooms that meet strict criteria (Gender + Lease Duration)")
    print("  💡 Other Close Matches: Similar rooms that might still work with flexibility")
    print("  ⭐ Lower scores indicate better matches")
    print("  🎯 Weighted scoring prioritizes important dimensions")


if __name__ == "__main__":
    main()
