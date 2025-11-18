from google.cloud.firestore import AsyncClient
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from datetime import date
import logging
import time

from app.db.firestore import get_firestore
from app.models.user import UserFilter

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(
            self
    ):
        self._firestore: AsyncClient = get_firestore()
    
    def _matches_filters(self, room_data: dict, user: UserFilter) -> bool:
        """
        Apply filters client-side.
        Returns True if room matches all filters.
        """
        # Location filter
        if user.location:
            if room_data.get('location') != user.location:
                return False
        
        # Max rent filter
        if user.max_rent is not None:
            room_rent = room_data.get('rent')
            if room_rent is None or room_rent > user.max_rent:
                return False
        
        # Room type filter
        if user.room_type:
            if room_data.get('room_type') != user.room_type:
                return False
        
        # Flatmate gender filter
        if user.flatmate_gender:
            if room_data.get('flatmate_gender') != user.flatmate_gender:
                return False
        # Bathroom filter
        if user.attached_bathroom:
            if room_data.get('attached_bathroom') != user.attached_bathroom:
                return False
        # Lease duration filter
        if user.lease_duration_months is not None:
            room_lease = room_data.get('lease_duration_months')
            if room_lease is None or room_lease > user.lease_duration_months:
                return False
        # Available from filter
        if user.available_from is not None:
            room_available = room_data.get('available_from')
            if room_available:
                # Handle both date and Firestore timestamp
                if isinstance(room_available, date):
                    if room_available > user.available_from:
                        return False
                elif hasattr(room_available, 'date'):
                    if room_available.date() > user.available_from:
                        return False
        return True
    
    async def find_best_match(
            self,
            user: UserFilter,
            limit: int = 10
    ):
        start_time = time.time()
        try:
            user_ref = self._firestore.collection('users').document(user.user_id)
            user_doc = await user_ref.get()
            
            if not user_doc.exists:
                raise ValueError(f"User {user.user_id} not found")
            
            user_data = user_doc.to_dict()
            
            if 'user_vector' not in user_data:
                raise ValueError(f"User {user.user_id} does not have user_vector")
            query_vector = user_data['user_vector']
            has_filters = user.has_filters()
            fetch_limit = user.limit * 5 if has_filters else user.limit
            fetch_limit = min(fetch_limit, 1000)  # Cap at Firestore max
            logger.info(
                f"Vector search: user={user.user_id}, "
                f"has_filters={has_filters}, fetch_limit={fetch_limit}"
            )
            vector_query = self._firestore.collection('rooms').find_nearest(
                vector_field='room_vector',
                query_vector=query_vector,
                distance_measure=DistanceMeasure.EUCLIDEAN,
                limit=fetch_limit
            )
            results = []
            total_fetched = 0
            async for doc in vector_query.stream():
                total_fetched += 1
                room_data = doc.to_dict()
                
                # Apply filters
                if not self._matches_filters(room_data, user):
                    continue
                
                # Add to results (already sorted by similarity from Firestore)
                results.append({
                    'room_id': doc.id,
                    'room_data': room_data
                })
                
                # Early termination
                if len(results) >= user.limit:
                    break
            
            # Log metrics
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Search completed: query_time_ms={elapsed_ms}, "
                f"fetched={total_fetched}, returned={len(results)}"
            )
            
            # Alert if slow
            if elapsed_ms > 500:
                logger.warning(f"SLOW QUERY: {elapsed_ms}ms for user={user.user_id}")
            
            return {
                'user_id': user.user_id,
                'matches': results,
                'total_results': len(results)
            }
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to get matched rooms: {str(e)}", exc_info=True)
            raise

def get_recommendation_service() -> RecommendationService:
    """Create new instance per request"""
    return RecommendationService()