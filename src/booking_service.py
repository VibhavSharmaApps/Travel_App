"""
Booking Service Module
=====================

This module handles all booking-related operations for the travel bot.
It integrates with the data manager to provide flight and hotel booking
functionality using Excel data for the demo version.

The service provides:
- Flight search and booking
- Hotel search and booking
- Booking management (view, modify, cancel)
- Booking reference generation
- Integration with notification system

Author: Travel Bot Team
Date: 2024
"""

import logging
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from .models import Booking, User
from .database import db_manager
from .data_manager import data_manager

logger = logging.getLogger(__name__)

class BookingService:
    """
    Booking Service Class
    ====================
    
    Handles all booking operations including flight and hotel reservations.
    Uses Excel data from the data manager for demo purposes.
    
    Attributes:
        data_manager: Reference to the global data manager instance
    """
    
    def __init__(self):
        """
        Initialize the Booking Service
        
        This method sets up the booking service and ensures
        the data manager is properly initialized.
        """
        self.data_manager = data_manager
        logger.info("Booking Service initialized")
    
    def create_booking(self, user_id: int, booking_type: str, details: Dict[str, Any]) -> Booking:
        """
        Create a new booking in the database
        
        This method creates a booking record with the provided details
        and stores it in the database. It generates a unique booking
        reference and validates the booking data.
        
        Args:
            user_id (int): ID of the user making the booking
            booking_type (str): Type of booking ('flight', 'hotel', 'package')
            details (Dict[str, Any]): Booking details including flight/hotel info
            
        Returns:
            Booking: The created booking object
            
        Raises:
            ValueError: If booking type is invalid or required details are missing
            Exception: If database operation fails
        """
        try:
            logger.info(f"Creating {booking_type} booking for user {user_id}")
            
            # Validate booking type
            if booking_type not in ['flight', 'hotel', 'package']:
                raise ValueError(f"Invalid booking type: {booking_type}")
            
            # Generate unique booking reference
            booking_reference = self._generate_booking_reference(booking_type)
            
            # Create booking object with basic information
            booking = Booking(
                user_id=user_id,
                booking_type=booking_type,
                booking_reference=booking_reference,
                details=details,
                status="confirmed"  # Default status for new bookings
            )
            
            # Add travel dates if provided
            if 'departure_date' in details:
                booking.departure_date = datetime.fromisoformat(details['departure_date'])
            if 'return_date' in details:
                booking.return_date = datetime.fromisoformat(details['return_date'])
            
            # Add location information
            if 'origin' in details:
                booking.origin = details['origin']
            if 'destination' in details:
                booking.destination = details['destination']
            
            # Add price information
            if 'total_price' in details:
                booking.total_price = details['total_price']
            if 'currency' in details:
                booking.currency = details['currency']
            
            # Save to database
            with db_manager.get_session() as session:
                session.add(booking)
                session.commit()
                
            logger.info(f"Successfully created booking {booking_reference} for user {user_id}")
            return booking
                
        except Exception as e:
            logger.error(f"Error creating booking: {e}")
            raise
    
    def get_user_bookings(self, user_id: int) -> List[Booking]:
        """
        Get all bookings for a specific user
        
        This method retrieves all booking records for a user from the database,
        ordered by creation date (most recent first).
        
        Args:
            user_id (int): ID of the user whose bookings to retrieve
            
        Returns:
            List[Booking]: List of booking objects for the user
            
        Note:
            Returns empty list if no bookings found or if error occurs
        """
        try:
            with db_manager.get_session() as session:
                bookings = session.query(Booking).filter(
                    Booking.user_id == user_id
                ).order_by(Booking.created_at.desc()).all()
                
            logger.info(f"Retrieved {len(bookings)} bookings for user {user_id}")
            return bookings
                
        except Exception as e:
            logger.error(f"Error getting user bookings: {e}")
            return []
    
    def get_booking_by_reference(self, booking_reference: str) -> Optional[Booking]:
        """
        Get a specific booking by its reference number
        
        This method searches for a booking using its unique reference number.
        Useful for checking booking status or retrieving booking details.
        
        Args:
            booking_reference (str): The booking reference number to search for
            
        Returns:
            Optional[Booking]: The booking object if found, None otherwise
        """
        try:
            with db_manager.get_session() as session:
                booking = session.query(Booking).filter(
                    Booking.booking_reference == booking_reference
                ).first()
                
            if booking:
                logger.info(f"Found booking {booking_reference}")
            else:
                logger.info(f"Booking {booking_reference} not found")
                
            return booking
                
        except Exception as e:
            logger.error(f"Error getting booking by reference: {e}")
            return None
    
    def update_booking(self, booking_reference: str, updates: Dict[str, Any]) -> Optional[Booking]:
        """
        Update an existing booking with new information
        
        This method allows updating various fields of an existing booking.
        Common updates include status changes, date modifications, or
        additional details.
        
        Args:
            booking_reference (str): The booking reference to update
            updates (Dict[str, Any]): Dictionary of fields to update
            
        Returns:
            Optional[Booking]: Updated booking object if successful, None otherwise
        """
        try:
            with db_manager.get_session() as session:
                booking = session.query(Booking).filter(
                    Booking.booking_reference == booking_reference
                ).first()
                
                if not booking:
                    logger.warning(f"Booking {booking_reference} not found for update")
                    return None
                
                # Update fields that exist in the booking model
                for key, value in updates.items():
                    if hasattr(booking, key):
                        setattr(booking, key, value)
                
                # Update the timestamp
                booking.updated_at = datetime.utcnow()
                session.commit()
                
            logger.info(f"Successfully updated booking {booking_reference}")
            return booking
                
        except Exception as e:
            logger.error(f"Error updating booking: {e}")
            return None
    
    def cancel_booking(self, booking_reference: str) -> bool:
        """
        Cancel a booking by changing its status to 'cancelled'
        
        This method marks a booking as cancelled rather than deleting it,
        which preserves the booking history and allows for potential
        reactivation or refund processing.
        
        Args:
            booking_reference (str): The booking reference to cancel
            
        Returns:
            bool: True if booking was successfully cancelled, False otherwise
        """
        try:
            with db_manager.get_session() as session:
                booking = session.query(Booking).filter(
                    Booking.booking_reference == booking_reference
                ).first()
                
                if not booking:
                    logger.warning(f"Booking {booking_reference} not found for cancellation")
                    return False
                
                # Update status to cancelled
                booking.status = "cancelled"
                booking.updated_at = datetime.utcnow()
                session.commit()
                
            logger.info(f"Successfully cancelled booking {booking_reference}")
            return True
                
        except Exception as e:
            logger.error(f"Error cancelling booking: {e}")
            return False
    
    def search_flights(self, origin: str = None, destination: str = None, 
                      date: str = None, passengers: int = 1, max_price: float = None) -> List[Dict]:
        """
        Search for available flights based on criteria
        
        This method searches the Excel flight data for flights matching
        the specified criteria. It uses the data manager to perform the search
        and returns formatted flight information.
        
        Args:
            origin (str, optional): Origin airport/city
            destination (str, optional): Destination airport/city
            date (str, optional): Travel date (for future enhancement)
            passengers (int): Number of passengers (default: 1)
            max_price (float, optional): Maximum price filter
            
        Returns:
            List[Dict]: List of matching flights with detailed information
        """
        try:
            # Use data manager to search flights
            flights = self.data_manager.search_flights(
                origin=origin,
                destination=destination,
                date=date,
                max_price=max_price,
                min_seats=passengers
            )
            
            logger.info(f"Found {len(flights)} flights matching search criteria")
            return flights
            
        except Exception as e:
            logger.error(f"Error searching flights: {e}")
            return []
    
    def search_hotels(self, location: str = None, check_in: str = None, 
                     check_out: str = None, guests: int = 1, max_price: float = None,
                     min_rating: float = None, amenities: List[str] = None) -> List[Dict]:
        """
        Search for available hotels based on criteria
        
        This method searches the Excel hotel data for hotels matching
        the specified criteria. It uses the data manager to perform the search
        and returns formatted hotel information.
        
        Args:
            location (str, optional): Hotel location/city
            check_in (str, optional): Check-in date (for future enhancement)
            check_out (str, optional): Check-out date (for future enhancement)
            guests (int): Number of guests (default: 1)
            max_price (float, optional): Maximum price per night
            min_rating (float, optional): Minimum hotel rating
            amenities (List[str], optional): Required amenities
            
        Returns:
            List[Dict]: List of matching hotels with detailed information
        """
        try:
            # Use data manager to search hotels
            hotels = self.data_manager.search_hotels(
                location=location,
                max_price=max_price,
                min_rating=min_rating,
                amenities=amenities
            )
            
            logger.info(f"Found {len(hotels)} hotels matching search criteria")
            return hotels
            
        except Exception as e:
            logger.error(f"Error searching hotels: {e}")
            return []
    
    def get_flight_details(self, flight_number: str) -> Optional[Dict]:
        """
        Get detailed information about a specific flight
        
        This method retrieves complete information about a flight
        using its flight number from the Excel data.
        
        Args:
            flight_number (str): The flight number to search for
            
        Returns:
            Optional[Dict]: Flight details if found, None otherwise
        """
        try:
            flight = self.data_manager.get_flight_by_number(flight_number)
            if flight:
                logger.info(f"Retrieved details for flight {flight_number}")
            else:
                logger.info(f"Flight {flight_number} not found")
            return flight
            
        except Exception as e:
            logger.error(f"Error getting flight details: {e}")
            return None
    
    def get_hotel_details(self, hotel_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific hotel
        
        This method retrieves complete information about a hotel
        using its name from the Excel data.
        
        Args:
            hotel_name (str): The hotel name to search for
            
        Returns:
            Optional[Dict]: Hotel details if found, None otherwise
        """
        try:
            hotel = self.data_manager.get_hotel_by_name(hotel_name)
            if hotel:
                logger.info(f"Retrieved details for hotel {hotel_name}")
            else:
                logger.info(f"Hotel {hotel_name} not found")
            return hotel
            
        except Exception as e:
            logger.error(f"Error getting hotel details: {e}")
            return None
    
    def get_available_airports(self) -> List[str]:
        """
        Get list of all available airports from the flight data
        
        This method returns a list of all unique airports that appear
        in the flight data (both as origins and destinations).
        
        Returns:
            List[str]: List of available airports
        """
        try:
            airports = self.data_manager.get_available_airports()
            logger.info(f"Retrieved {len(airports)} available airports")
            return airports
            
        except Exception as e:
            logger.error(f"Error getting available airports: {e}")
            return []
    
    def get_available_locations(self) -> List[str]:
        """
        Get list of all available hotel locations from the hotel data
        
        This method returns a list of all unique locations where
        hotels are available in the hotel data.
        
        Returns:
            List[str]: List of available hotel locations
        """
        try:
            locations = self.data_manager.get_available_locations()
            logger.info(f"Retrieved {len(locations)} available hotel locations")
            return locations
            
        except Exception as e:
            logger.error(f"Error getting available locations: {e}")
            return []
    
    def get_booking_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about bookings and available data
        
        This method provides comprehensive statistics about the booking
        system, including data availability and booking metrics.
        
        Returns:
            Dict[str, Any]: Statistics about bookings and data
        """
        try:
            # Get data statistics from data manager
            data_stats = self.data_manager.get_statistics()
            
            # Get booking statistics from database
            with db_manager.get_session() as session:
                total_bookings = session.query(Booking).count()
                confirmed_bookings = session.query(Booking).filter(
                    Booking.status == "confirmed"
                ).count()
                cancelled_bookings = session.query(Booking).filter(
                    Booking.status == "cancelled"
                ).count()
            
            stats = {
                'data': data_stats,
                'bookings': {
                    'total': total_bookings,
                    'confirmed': confirmed_bookings,
                    'cancelled': cancelled_bookings
                }
            }
            
            logger.info("Retrieved booking statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting booking statistics: {e}")
            return {}
    
    def _generate_booking_reference(self, booking_type: str) -> str:
        """
        Generate a unique booking reference number
        
        This method creates a unique booking reference based on the
        booking type. The reference format is:
        - FL + 6 digits for flights
        - HT + 6 digits for hotels
        - PK + 6 digits for packages
        
        Args:
            booking_type (str): Type of booking ('flight', 'hotel', 'package')
            
        Returns:
            str: Unique booking reference number
        """
        # Define prefix based on booking type
        prefix_map = {
            'flight': 'FL',
            'hotel': 'HT',
            'package': 'PK'
        }
        
        prefix = prefix_map.get(booking_type, 'BK')
        
        # Generate 6-digit random number
        random_number = random.randint(100000, 999999)
        
        booking_reference = f"{prefix}{random_number}"
        
        logger.debug(f"Generated booking reference: {booking_reference}")
        return booking_reference

# Global booking service instance
booking_service = BookingService() 