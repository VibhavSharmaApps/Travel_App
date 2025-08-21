"""
Data Manager Module
==================

This module handles loading and processing of travel data from Excel files.
It provides a centralized interface for accessing flight and hotel information
stored in Excel format for the demo version of the travel bot.

The module supports:
- Loading flight data from Excel files
- Loading hotel data from Excel files
- Filtering and searching data based on criteria
- Data validation and error handling
- Caching for performance optimization

Author: Travel Bot Team
Date: 2024
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class DataManager:
    """
    Data Manager Class
    =================
    
    Manages travel data from Excel files including flights and hotels.
    Provides methods for loading, filtering, and searching travel data.
    
    Attributes:
        flights_data (pd.DataFrame): DataFrame containing flight information
        hotels_data (pd.DataFrame): DataFrame containing hotel information
        data_loaded (bool): Flag indicating if data has been loaded
        cache (Dict): Cache for storing frequently accessed data
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Data Manager
        
        Args:
            data_dir (str): Directory containing Excel data files
        """
        self.data_dir = Path(data_dir)
        self.flights_data = None
        self.hotels_data = None
        self.data_loaded = False
        self.cache = {}
        
        # Define expected column names for validation
        self.expected_flight_columns = [
            'Flight Number', 'Airline', 'Origin', 'Destination', 
            'Departure Time', 'Arrival Time', 'Duration', 'Price', 
            'Seats Available', 'Stops', 'Aircraft', 'Gate', 'Status'
        ]
        
        self.expected_hotel_columns = [
            'Hotel Name', 'Location', 'Rating', 'Price Per Night',
            'Amenities', 'Room Types', 'Available Rooms', 
            'Check-in', 'Check-out', 'Address', 'Phone', 'Website'
        ]
        
        logger.info("Data Manager initialized")
    
    def load_data(self) -> bool:
        """
        Load flight and hotel data from Excel files
        
        Returns:
            bool: True if data loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If Excel files are not found
            ValueError: If Excel files have invalid structure
        """
        try:
            logger.info("Loading travel data from Excel files...")
            
            # Load flight data
            flights_file = self.data_dir / "flights.xlsx"
            if not flights_file.exists():
                raise FileNotFoundError(f"Flight data file not found: {flights_file}")
            
            self.flights_data = pd.read_excel(flights_file, engine='openpyxl')
            logger.info(f"Loaded {len(self.flights_data)} flight records")
            logger.info(f"Actual columns in flights file: {list(self.flights_data.columns)}")
            
            # Validate flight data structure
            self._validate_flight_data()
            
            # Load hotel data
            hotels_file = self.data_dir / "hotels.xlsx"
            if not hotels_file.exists():
                raise FileNotFoundError(f"Hotel data file not found: {hotels_file}")
            
            self.hotels_data = pd.read_excel(hotels_file, engine='openpyxl')
            logger.info(f"Loaded {len(self.hotels_data)} hotel records")
            logger.info(f"Actual columns in hotels file: {list(self.hotels_data.columns)}")
            
            # Validate hotel data structure
            self._validate_hotel_data()
            
            # Clean and preprocess data
            self._clean_data()
            
            self.data_loaded = True
            logger.info("Travel data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading travel data: {e}")
            return False
    
    def _validate_flight_data(self):
        """
        Validate flight data structure and content
        
        Raises:
            ValueError: If flight data is invalid
        """
        if self.flights_data is None:
            raise ValueError("Flight data is None")
        
        # Check if all expected columns exist
        missing_columns = set(self.expected_flight_columns) - set(self.flights_data.columns)
        if missing_columns:
            raise ValueError(f"Missing flight columns: {missing_columns}")
        
        # Check for empty data
        if self.flights_data.empty:
            raise ValueError("Flight data is empty")
        
        # Validate data types and content
        for column in ['Seats Available', 'Stops']:
            if not pd.api.types.is_numeric_dtype(self.flights_data[column]):
                # Try to convert to numeric
                try:
                    self.flights_data[column] = pd.to_numeric(self.flights_data[column], errors='coerce')
                    logger.info(f"Converted column '{column}' to numeric")
                except Exception as e:
                    raise ValueError(f"Column '{column}' must be numeric and could not be converted: {e}")
        
        logger.info("Flight data validation passed")
    
    def _validate_hotel_data(self):
        """
        Validate hotel data structure and content
        
        Raises:
            ValueError: If hotel data is invalid
        """
        if self.hotels_data is None:
            raise ValueError("Hotel data is None")
        
        # Check if all expected columns exist
        missing_columns = set(self.expected_hotel_columns) - set(self.hotels_data.columns)
        if missing_columns:
            raise ValueError(f"Missing hotel columns: {missing_columns}")
        
        # Check for empty data
        if self.hotels_data.empty:
            raise ValueError("Hotel data is empty")
        
        # Validate rating column
        if not pd.api.types.is_numeric_dtype(self.hotels_data['Rating']):
            raise ValueError("Rating column must be numeric")
        
        # Validate price column (remove $ and convert to numeric)
        price_series = self.hotels_data['Price Per Night'].astype(str).str.replace('$', '').str.replace(',', '')
        if not all(price_series.str.replace('.', '').str.isdigit()):
            raise ValueError("Price Per Night column contains invalid values")
        
        logger.info("Hotel data validation passed")
    
    def _clean_data(self):
        """
        Clean and preprocess the loaded data
        
        This method:
        - Removes leading/trailing whitespace
        - Converts price strings to numeric values
        - Standardizes time formats
        - Handles missing values
        """
        logger.info("Cleaning and preprocessing data...")
        
        # Clean flight data
        if self.flights_data is not None:
            # Remove whitespace from string columns
            string_columns = ['Flight Number', 'Airline', 'Origin', 'Destination', 'Aircraft', 'Gate', 'Status']
            for col in string_columns:
                if col in self.flights_data.columns:
                    self.flights_data[col] = self.flights_data[col].astype(str).str.strip()
            
            # Convert price to numeric (remove $ and commas)
            if 'Price' in self.flights_data.columns:
                self.flights_data['Price'] = self.flights_data['Price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        
        # Clean hotel data
        if self.hotels_data is not None:
            # Remove whitespace from string columns
            string_columns = ['Hotel Name', 'Location', 'Address', 'Phone', 'Website']
            for col in string_columns:
                if col in self.hotels_data.columns:
                    self.hotels_data[col] = self.hotels_data[col].astype(str).str.strip()
            
            # Convert price to numeric
            if 'Price Per Night' in self.hotels_data.columns:
                self.hotels_data['Price Per Night'] = self.hotels_data['Price Per Night'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
            
            # Convert amenities and room types to lists
            if 'Amenities' in self.hotels_data.columns:
                self.hotels_data['Amenities'] = self.hotels_data['Amenities'].apply(
                    lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else []
                )
            
            if 'Room Types' in self.hotels_data.columns:
                self.hotels_data['Room Types'] = self.hotels_data['Room Types'].apply(
                    lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else []
                )
        
        logger.info("Data cleaning completed")
    
    def search_flights(self, origin: str = None, destination: str = None, 
                      date: str = None, max_price: float = None, 
                      min_seats: int = None) -> List[Dict]:
        """
        Search for flights based on specified criteria
        
        Args:
            origin (str, optional): Origin airport/city
            destination (str, optional): Destination airport/city
            date (str, optional): Travel date (for future enhancement)
            max_price (float, optional): Maximum price filter
            min_seats (int, optional): Minimum available seats
            
        Returns:
            List[Dict]: List of matching flights
            
        Note:
            This is a simplified search. In a real implementation,
            date filtering would be implemented based on actual flight schedules.
        """
        if not self.data_loaded or self.flights_data is None:
            logger.warning("Flight data not loaded")
            return []
        
        # Create a copy for filtering
        filtered_data = self.flights_data.copy()
        
        # Apply filters
        if origin:
            filtered_data = filtered_data[
                filtered_data['Origin'].str.contains(origin, case=False, na=False)
            ]
        
        if destination:
            filtered_data = filtered_data[
                filtered_data['Destination'].str.contains(destination, case=False, na=False)
            ]
        
        if max_price is not None:
            filtered_data = filtered_data[filtered_data['Price'] <= max_price]
        
        if min_seats is not None:
            filtered_data = filtered_data[filtered_data['Seats Available'] >= min_seats]
        
        # Convert to list of dictionaries
        flights = filtered_data.to_dict('records')
        
        logger.info(f"Found {len(flights)} flights matching criteria")
        return flights
    
    def search_hotels(self, location: str = None, max_price: float = None,
                     min_rating: float = None, amenities: List[str] = None,
                     room_type: str = None) -> List[Dict]:
        """
        Search for hotels based on specified criteria
        
        Args:
            location (str, optional): Hotel location/city
            max_price (float, optional): Maximum price per night
            min_rating (float, optional): Minimum rating
            amenities (List[str], optional): Required amenities
            room_type (str, optional): Required room type
            
        Returns:
            List[Dict]: List of matching hotels
        """
        if not self.data_loaded or self.hotels_data is None:
            logger.warning("Hotel data not loaded")
            return []
        
        # Create a copy for filtering
        filtered_data = self.hotels_data.copy()
        
        # Apply filters
        if location:
            filtered_data = filtered_data[
                filtered_data['Location'].str.contains(location, case=False, na=False)
            ]
        
        if max_price is not None:
            filtered_data = filtered_data[filtered_data['Price Per Night'] <= max_price]
        
        if min_rating is not None:
            filtered_data = filtered_data[filtered_data['Rating'] >= min_rating]
        
        if amenities:
            # Filter hotels that have all required amenities
            for amenity in amenities:
                filtered_data = filtered_data[
                    filtered_data['Amenities'].apply(
                        lambda x: amenity.lower() in [a.lower() for a in x]
                    )
                ]
        
        if room_type:
            # Filter hotels that have the required room type
            filtered_data = filtered_data[
                filtered_data['Room Types'].apply(
                    lambda x: room_type.lower() in [r.lower() for r in x]
                )
            ]
        
        # Convert to list of dictionaries
        hotels = filtered_data.to_dict('records')
        
        logger.info(f"Found {len(hotels)} hotels matching criteria")
        return hotels
    
    def get_flight_by_number(self, flight_number: str) -> Optional[Dict]:
        """
        Get specific flight by flight number
        
        Args:
            flight_number (str): Flight number to search for
            
        Returns:
            Dict: Flight information or None if not found
        """
        if not self.data_loaded or self.flights_data is None:
            return None
        
        flight = self.flights_data[
            self.flights_data['Flight Number'].str.contains(flight_number, case=False, na=False)
        ]
        
        if flight.empty:
            return None
        
        return flight.iloc[0].to_dict()
    
    def get_hotel_by_name(self, hotel_name: str) -> Optional[Dict]:
        """
        Get specific hotel by name
        
        Args:
            hotel_name (str): Hotel name to search for
            
        Returns:
            Dict: Hotel information or None if not found
        """
        if not self.data_loaded or self.hotels_data is None:
            return None
        
        hotel = self.hotels_data[
            self.hotels_data['Hotel Name'].str.contains(hotel_name, case=False, na=False)
        ]
        
        if hotel.empty:
            return None
        
        return hotel.iloc[0].to_dict()
    
    def get_available_airports(self) -> List[str]:
        """
        Get list of all available airports from flight data
        
        Returns:
            List[str]: List of unique airports
        """
        if not self.data_loaded or self.flights_data is None:
            return []
        
        origins = set(self.flights_data['Origin'].dropna())
        destinations = set(self.flights_data['Destination'].dropna())
        all_airports = list(origins.union(destinations))
        
        return sorted(all_airports)
    
    def get_available_locations(self) -> List[str]:
        """
        Get list of all available hotel locations
        
        Returns:
            List[str]: List of unique locations
        """
        if not self.data_loaded or self.hotels_data is None:
            return []
        
        locations = list(self.hotels_data['Location'].dropna().unique())
        return sorted(locations)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data
        
        Returns:
            Dict[str, Any]: Statistics about flights and hotels
        """
        stats = {
            'flights': {
                'total': 0,
                'airlines': [],
                'airports': [],
                'price_range': {'min': 0, 'max': 0}
            },
            'hotels': {
                'total': 0,
                'locations': [],
                'rating_range': {'min': 0, 'max': 0},
                'price_range': {'min': 0, 'max': 0}
            }
        }
        
        if self.data_loaded:
            # Flight statistics
            if self.flights_data is not None:
                stats['flights']['total'] = len(self.flights_data)
                stats['flights']['airlines'] = list(self.flights_data['Airline'].unique())
                stats['flights']['airports'] = self.get_available_airports()
                if 'Price' in self.flights_data.columns:
                    stats['flights']['price_range'] = {
                        'min': float(self.flights_data['Price'].min()),
                        'max': float(self.flights_data['Price'].max())
                    }
            
            # Hotel statistics
            if self.hotels_data is not None:
                stats['hotels']['total'] = len(self.hotels_data)
                stats['hotels']['locations'] = self.get_available_locations()
                if 'Rating' in self.hotels_data.columns:
                    stats['hotels']['rating_range'] = {
                        'min': float(self.hotels_data['Rating'].min()),
                        'max': float(self.hotels_data['Rating'].max())
                    }
                if 'Price Per Night' in self.hotels_data.columns:
                    stats['hotels']['price_range'] = {
                        'min': float(self.hotels_data['Price Per Night'].min()),
                        'max': float(self.hotels_data['Price Per Night'].max())
                    }
        
        return stats

# Global data manager instance
data_manager = DataManager() 