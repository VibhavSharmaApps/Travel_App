import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.ai_engine import AIEngine

class TestAIEngine:
    """Test cases for AI Engine"""
    
    @pytest.fixture
    def ai_engine(self):
        """Create AI engine instance for testing"""
        return AIEngine()
    
    def test_intent_classification(self, ai_engine):
        """Test intent classification"""
        # Test flight booking intent
        intent, confidence = ai_engine.classify_intent("I want to book a flight")
        assert intent == "book_flight"
        assert confidence > 0.5
        
        # Test hotel booking intent
        intent, confidence = ai_engine.classify_intent("I need a hotel")
        assert intent == "book_hotel"
        assert confidence > 0.5
        
        # Test general help intent
        intent, confidence = ai_engine.classify_intent("help me")
        assert intent == "general_help"
        assert confidence > 0.5
    
    def test_entity_extraction(self, ai_engine):
        """Test entity extraction"""
        # Test date extraction
        entities = ai_engine.extract_entities("I need a flight on March 15th")
        assert "dates" in entities
        
        # Test location extraction
        entities = ai_engine.extract_entities("I want to fly from New York to London")
        assert "locations" in entities
        
        # Test number extraction
        entities = ai_engine.extract_entities("I need 2 passengers")
        assert "numbers" in entities
        
        # Test booking reference extraction
        entities = ai_engine.extract_entities("Check my booking FL123456")
        assert "booking_reference" in entities
        assert entities["booking_reference"] == "FL123456"
    
    def test_message_processing(self, ai_engine):
        """Test complete message processing"""
        # Test flight booking message
        result = ai_engine.process_message("I need a flight from New York to London on March 15th for 2 passengers")
        assert result["intent"] == "book_flight"
        assert "response" in result
        assert "entities" in result
        assert "confidence" in result
        
        # Test hotel booking message
        result = ai_engine.process_message("I want to book a hotel in Paris from March 15th to March 20th")
        assert result["intent"] == "book_hotel"
        assert "response" in result
    
    def test_error_handling(self, ai_engine):
        """Test error handling"""
        # Test with empty message
        result = ai_engine.process_message("")
        assert result["intent"] == "error" or result["intent"] == "general_help"
        
        # Test with None message
        result = ai_engine.process_message(None)
        assert result["intent"] == "error" or result["intent"] == "general_help"
    
    def test_confidence_scores(self, ai_engine):
        """Test confidence score ranges"""
        # Test various messages and ensure confidence is between 0 and 1
        test_messages = [
            "book a flight",
            "reserve hotel",
            "check my booking",
            "cancel reservation",
            "weather update",
            "random message"
        ]
        
        for message in test_messages:
            intent, confidence = ai_engine.classify_intent(message)
            assert 0 <= confidence <= 1
            assert intent in [
                "book_flight", "book_hotel", "check_booking", 
                "modify_booking", "cancel_booking", "travel_updates",
                "weather_info", "general_help"
            ] 