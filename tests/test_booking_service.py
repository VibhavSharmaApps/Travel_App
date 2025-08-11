import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.booking_service import BookingService, MockFlightAPI, MockHotelAPI
from src.models import User, Booking

class TestBookingService:
    """Test cases for Booking Service"""
    
    @pytest.fixture
    def booking_service(self):
        """Create booking service instance for testing"""
        return BookingService()
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user for testing"""
        return User(
            telegram_id=12345,
            username="testuser",
            first_name="Test",
            last_name="User"
        )
    
    def test_flight_search(self, booking_service):
        """Test flight search functionality"""
        flights = booking_service.search_flights(
            origin="New York",
            destination="London",
            date="2024-03-15",
            passengers=2
        )
        
        assert isinstance(flights, list)
        assert len(flights) > 0
        
        # Check flight structure
        flight = flights[0]
        assert "flight_number" in flight
        assert "airline" in flight
        assert "origin" in flight
        assert "destination" in flight
        assert "departure_time" in flight
        assert "arrival_time" in flight
        assert "price" in flight
    
    def test_hotel_search(self, booking_service):
        """Test hotel search functionality"""
        hotels = booking_service.search_hotels(
            location="Paris",
            check_in="2024-03-15",
            check_out="2024-03-20",
            guests=2
        )
        
        assert isinstance(hotels, list)
        assert len(hotels) > 0
        
        # Check hotel structure
        hotel = hotels[0]
        assert "hotel_name" in hotel
        assert "location" in hotel
        assert "rating" in hotel
        assert "price_per_night" in hotel
        assert "amenities" in hotel
        assert "room_types" in hotel
    
    def test_booking_reference_generation(self, booking_service):
        """Test booking reference generation"""
        # Test flight booking reference
        flight_ref = booking_service._generate_booking_reference("flight")
        assert flight_ref.startswith("FL")
        assert len(flight_ref) == 8  # FL + 6 digits
        
        # Test hotel booking reference
        hotel_ref = booking_service._generate_booking_reference("hotel")
        assert hotel_ref.startswith("HT")
        assert len(hotel_ref) == 8  # HT + 6 digits
        
        # Test package booking reference
        package_ref = booking_service._generate_booking_reference("package")
        assert package_ref.startswith("PK")
        assert len(package_ref) == 8  # PK + 6 digits

class TestMockFlightAPI:
    """Test cases for Mock Flight API"""
    
    @pytest.fixture
    def flight_api(self):
        """Create mock flight API instance for testing"""
        return MockFlightAPI()
    
    def test_flight_search_results(self, flight_api):
        """Test flight search returns expected results"""
        flights = flight_api.search_flights(
            origin="New York",
            destination="London",
            date="2024-03-15",
            passengers=2
        )
        
        assert len(flights) == 5  # Should return 5 mock flights
        
        for flight in flights:
            assert flight["origin"] == "New York"
            assert flight["destination"] == "London"
            assert flight["airline"] in ["Delta", "American", "United", "Southwest", "JetBlue"]
            assert flight["seats_available"] > 0
            assert flight["stops"] in [0, 1, 2]
    
    def test_flight_data_structure(self, flight_api):
        """Test flight data structure is correct"""
        flights = flight_api.search_flights("LAX", "JFK", "2024-03-15", 1)
        flight = flights[0]
        
        # Check required fields
        required_fields = [
            "flight_number", "airline", "origin", "destination",
            "departure_time", "arrival_time", "duration", "price",
            "seats_available", "stops"
        ]
        
        for field in required_fields:
            assert field in flight
        
        # Check data types
        assert isinstance(flight["flight_number"], str)
        assert isinstance(flight["airline"], str)
        assert isinstance(flight["seats_available"], int)
        assert isinstance(flight["stops"], int)
        assert isinstance(flight["price"], str)

class TestMockHotelAPI:
    """Test cases for Mock Hotel API"""
    
    @pytest.fixture
    def hotel_api(self):
        """Create mock hotel API instance for testing"""
        return MockHotelAPI()
    
    def test_hotel_search_results(self, hotel_api):
        """Test hotel search returns expected results"""
        hotels = hotel_api.search_hotels(
            location="Paris",
            check_in="2024-03-15",
            check_out="2024-03-20",
            guests=2
        )
        
        assert len(hotels) == 5  # Should return 5 mock hotels
        
        for hotel in hotels:
            assert "Paris" in hotel["hotel_name"]
            assert hotel["location"] == "Paris"
            assert hotel["check_in"] == "2024-03-15"
            assert hotel["check_out"] == "2024-03-20"
    
    def test_hotel_data_structure(self, hotel_api):
        """Test hotel data structure is correct"""
        hotels = hotel_api.search_hotels("London", "2024-03-15", "2024-03-20", 2)
        hotel = hotels[0]
        
        # Check required fields
        required_fields = [
            "hotel_name", "location", "rating", "price_per_night",
            "amenities", "room_types", "available_rooms",
            "check_in", "check_out"
        ]
        
        for field in required_fields:
            assert field in hotel
        
        # Check data types
        assert isinstance(hotel["hotel_name"], str)
        assert isinstance(hotel["rating"], float)
        assert isinstance(hotel["amenities"], list)
        assert isinstance(hotel["room_types"], list)
        assert isinstance(hotel["available_rooms"], int)
        
        # Check value ranges
        assert 3.0 <= hotel["rating"] <= 5.0
        assert hotel["available_rooms"] > 0
        assert len(hotel["amenities"]) >= 2
        assert len(hotel["room_types"]) >= 1 