"""
AI Engine Module
===============

This module provides natural language processing capabilities for the travel bot
using Hugging Face transformers. It handles intent classification, entity extraction,
and response generation for travel-related queries.

The module integrates with:
- Hugging Face transformers for NLP tasks
- Sentence transformers for semantic similarity
- Custom travel datasets for domain-specific understanding
- Excel data for context-aware responses

Key Features:
- Intent classification for travel queries
- Entity extraction (dates, locations, numbers, booking references)
- Context-aware response generation
- Confidence scoring for predictions
- Integration with travel data for personalized responses

Author: Travel Bot Team
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import re
from datetime import datetime
import torch
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from .config import config
from .data_manager import data_manager

logger = logging.getLogger(__name__)

class AIEngine:
    """
    AI Engine Class
    ===============
    
    Main AI engine for natural language processing in the travel domain.
    Uses Hugging Face models for intent classification, entity extraction,
    and response generation.
    
    Attributes:
        conversation_pipeline: Hugging Face pipeline for text generation
        sentence_encoder: Sentence transformer for semantic similarity
        intent_patterns: Dictionary of intent patterns for classification
        data_manager: Reference to data manager for context-aware responses
    """
    
    def __init__(self):
        """
        Initialize the AI Engine
        
        This method sets up all necessary models and components:
        - Loads Hugging Face models for conversation and encoding
        - Defines intent patterns for travel domain
        - Initializes data manager integration
        - Sets up confidence thresholds
        """
        self.conversation_pipeline = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.sentence_encoder = None
        self.data_manager = data_manager
        
        # Initialize models and patterns
        self._setup_models()
        self._setup_intent_patterns()
        
        logger.info("AI Engine initialized successfully")
    
    def _setup_models(self):
        """
        Setup Hugging Face models for NLP tasks
        
        This method initializes the required models:
        - Conversation model for response generation
        - Sentence encoder for intent classification
        - Entity extraction models (if available)
        
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info("Setting up Hugging Face models...")
            
            # Setup conversation model for response generation
            # Using BlenderBot for natural conversation flow
            self.conversation_pipeline = pipeline(
                "text2text-generation",
                model=config.HUGGINGFACE_MODEL_NAME,
                token=config.HUGGINGFACE_API_TOKEN,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Setup sentence encoder for intent classification
            # Using all-MiniLM-L6-v2 for fast and accurate semantic similarity
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Hugging Face models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup AI models: {e}")
            raise
    
    def _setup_intent_patterns(self):
        """
        Setup intent patterns for travel domain classification
        
        This method defines patterns and keywords for different travel intents.
        Each intent has multiple patterns to improve classification accuracy.
        """
        self.intent_patterns = {
            # Flight booking intents
            'book_flight': [
                'book a flight', 'reserve flight', 'flight booking', 'book flight',
                'need a flight', 'want to fly', 'schedule flight', 'find flights',
                'search flights', 'flight reservation', 'airline ticket',
                'fly from', 'fly to', 'departure', 'arrival'
            ],
            
            # Hotel booking intents
            'book_hotel': [
                'book hotel', 'reserve hotel', 'hotel booking', 'need hotel',
                'want hotel', 'find hotel', 'accommodation', 'room reservation',
                'stay at', 'hotel room', 'lodging', 'check in', 'check out'
            ],
            
            # Booking management intents
            'check_booking': [
                'check booking', 'my booking', 'booking status', 'reservation status',
                'flight status', 'hotel reservation', 'my reservation', 'booking details',
                'confirm booking', 'booking reference', 'reservation number'
            ],
            
            'modify_booking': [
                'modify booking', 'change booking', 'update booking', 'edit reservation',
                'change flight', 'modify hotel', 'reschedule', 'alter booking',
                'update reservation', 'change dates', 'modify dates'
            ],
            
            'cancel_booking': [
                'cancel booking', 'cancel flight', 'cancel hotel', 'cancel reservation',
                'refund', 'cancel trip', 'cancel ticket', 'cancel room',
                'cancel my booking', 'cancel reservation'
            ],
            
            # Travel information intents
            'travel_updates': [
                'flight updates', 'travel updates', 'status update', 'delay',
                'gate change', 'boarding', 'departure time', 'arrival time',
                'flight status', 'travel status', 'itinerary update'
            ],
            
            'weather_info': [
                'weather', 'weather forecast', 'temperature', 'climate',
                'weather at destination', 'weather in', 'weather conditions',
                'temperature at', 'climate in', 'weather report'
            ],
            
            # Additional travel services
            'baggage_info': [
                'baggage', 'luggage', 'baggage claim', 'lost luggage',
                'baggage tracking', 'carry on', 'checked baggage', 'baggage policy'
            ],
            
            'meal_preference': [
                'meal', 'food', 'dietary', 'vegetarian', 'vegan', 'meal preference',
                'special meal', 'dietary requirement', 'food allergy', 'meal request'
            ],
            
            'seat_preference': [
                'seat', 'seat assignment', 'window seat', 'aisle seat',
                'seat preference', 'seat selection', 'preferred seat', 'seat upgrade'
            ],
            
            # General help and support
            'general_help': [
                'help', 'support', 'assistance', 'how to', 'what can you do',
                'features', 'capabilities', 'guide', 'tutorial', 'instructions'
            ]
        }
        
        logger.info(f"Setup {len(self.intent_patterns)} intent patterns")
    
    def classify_intent(self, message: str) -> Tuple[str, float]:
        """
        Classify user intent from message using semantic similarity
        
        This method uses sentence transformers to calculate similarity between
        the user message and predefined intent patterns. It returns the most
        likely intent along with a confidence score.
        
        Args:
            message (str): User message to classify
            
        Returns:
            Tuple[str, float]: (intent, confidence_score)
            
        Note:
            Confidence scores range from 0.0 to 1.0, where 1.0 indicates
            perfect similarity with the intent pattern.
        """
        try:
            message_lower = message.lower()
            
            # Calculate similarity scores for each intent
            intent_scores = {}
            message_embedding = self.sentence_encoder.encode(message_lower)
            
            for intent, patterns in self.intent_patterns.items():
                # Encode all patterns for this intent
                pattern_embeddings = self.sentence_encoder.encode(patterns)
                
                # Calculate cosine similarity between message and patterns
                similarities = torch.cosine_similarity(
                    torch.tensor(message_embedding).unsqueeze(0),
                    torch.tensor(pattern_embeddings)
                )
                
                # Take the maximum similarity score for this intent
                intent_scores[intent] = float(torch.max(similarities))
            
            # Get the intent with highest confidence
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            logger.debug(f"Intent classification: {best_intent[0]} (confidence: {best_intent[1]:.3f})")
            return best_intent[0], best_intent[1]
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return "general_help", 0.5
    
    def extract_entities(self, message: str) -> Dict[str, Any]:
        """
        Extract entities from user message using pattern matching
        
        This method extracts various types of entities from the user message:
        - Dates and times
        - Locations (cities, airports)
        - Numbers (passengers, prices)
        - Booking references
        - Airlines and flight numbers
        
        Args:
            message (str): User message to extract entities from
            
        Returns:
            Dict[str, Any]: Dictionary of extracted entities
        """
        entities = {}
        
        try:
            # Extract dates using various patterns
            date_patterns = [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
                r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                r'(tomorrow|today|next week|next month|next year)',  # Relative dates
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',  # Month day
                r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))',  # Day month
                r'(\d{1,2}:\d{2}\s*(am|pm)?)',  # Time patterns
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, message.lower())
                if matches:
                    entities['dates'] = matches
            
            # Extract locations (cities, airports)
            location_patterns = [
                r'from\s+([A-Za-z\s]+?)(?:\s+to|\s+on|\s+for|$)',  # Origin
                r'to\s+([A-Za-z\s]+?)(?:\s+on|\s+for|$)',  # Destination
                r'in\s+([A-Za-z\s]+?)(?:\s+for|\s+on|$)',  # Location
                r'at\s+([A-Za-z\s]+?)(?:\s+for|\s+on|$)',  # Specific location
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, message.lower())
                if matches:
                    entities['locations'] = matches
            
            # Extract numbers (passengers, price, etc.)
            number_patterns = [
                r'(\d+)\s+(passenger|person|people|guest|adult|child|infant)',  # Passenger count
                r'(\d+)\s+(dollar|euro|pound|usd|eur|gbp)',  # Currency amounts
                r'(\$\d+(?:,\d{3})*(?:\.\d{2})?)',  # Dollar amounts
                r'(\d+)\s+(room|bed|night|day)',  # Room/night count
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, message.lower())
                if matches:
                    entities['numbers'] = matches
            
            # Extract booking references (common formats)
            booking_ref_patterns = [
                r'([A-Z]{2,3}\d{6,10})',  # Airline codes + numbers
                r'(FL\d{6})',  # Flight booking format
                r'(HT\d{6})',  # Hotel booking format
                r'(PK\d{6})',  # Package booking format
            ]
            
            for pattern in booking_ref_patterns:
                booking_matches = re.findall(pattern, message.upper())
                if booking_matches:
                    entities['booking_reference'] = booking_matches[0]
                    break
            
            # Extract airlines and flight numbers
            airline_patterns = [
                r'((?:delta|american|united|southwest|jetblue|alaska|spirit|frontier)\s+airlines?)',
                r'((?:delta|american|united|southwest|jetblue|alaska|spirit|frontier))',
                r'flight\s+([A-Z]{2}\d{3,4})',  # Flight numbers
            ]
            
            for pattern in airline_patterns:
                matches = re.findall(pattern, message.lower())
                if matches:
                    entities['airlines'] = matches
            
            logger.debug(f"Extracted entities: {entities}")
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return entities
    
    def generate_response(self, message: str, context: Optional[Dict] = None) -> str:
        """
        Generate AI response using Hugging Face model
        
        This method generates contextual responses based on the user message
        and conversation context. It uses the conversation pipeline to create
        natural, context-aware responses.
        
        Args:
            message (str): User message
            context (Optional[Dict]): Conversation context including intent and entities
            
        Returns:
            str: Generated response
            
        Note:
            The response generation can be enhanced with travel-specific
            information from the data manager for more personalized responses.
        """
        try:
            # Prepare context-aware prompt
            if context:
                intent = context.get('intent', 'general')
                entities = context.get('entities', {})
                
                # Create enhanced prompt with travel context
                prompt = self._create_enhanced_prompt(message, intent, entities)
            else:
                prompt = f"User: {message}\nAssistant:"
            
            # Generate response using Hugging Face pipeline
            response = self.conversation_pipeline(
                prompt,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            generated_response = response[0]['generated_text'].strip()
            
            # Post-process response for travel domain
            final_response = self._post_process_response(generated_response, context)
            
            logger.debug(f"Generated response: {final_response}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Please try again."
    
    def _create_enhanced_prompt(self, message: str, intent: str, entities: Dict) -> str:
        """
        Create enhanced prompt with travel context
        
        This method creates a more detailed prompt that includes travel-specific
        context to help the model generate more relevant responses.
        
        Args:
            message (str): User message
            intent (str): Classified intent
            entities (Dict): Extracted entities
            
        Returns:
            str: Enhanced prompt for response generation
        """
        # Base context
        context_parts = [f"Context: Travel assistant helping with {intent}"]
        
        # Add entity information
        if entities.get('locations'):
            context_parts.append(f"Locations: {', '.join(entities['locations'])}")
        
        if entities.get('dates'):
            context_parts.append(f"Dates: {', '.join(entities['dates'])}")
        
        if entities.get('numbers'):
            context_parts.append(f"Numbers: {', '.join(entities['numbers'])}")
        
        if entities.get('booking_reference'):
            context_parts.append(f"Booking: {entities['booking_reference']}")
        
        # Add travel data context if available
        if self.data_manager.data_loaded:
            stats = self.data_manager.get_statistics()
            context_parts.append(f"Available: {stats['flights']['total']} flights, {stats['hotels']['total']} hotels")
        
        context_str = " | ".join(context_parts)
        
        return f"{context_str}\nUser: {message}\nAssistant:"
    
    def _post_process_response(self, response: str, context: Optional[Dict] = None) -> str:
        """
        Post-process generated response for travel domain
        
        This method enhances the generated response with travel-specific
        information and formatting.
        
        Args:
            response (str): Raw generated response
            context (Optional[Dict]): Conversation context
            
        Returns:
            str: Enhanced response
        """
        # Add travel-specific enhancements
        if context and context.get('intent') == 'book_flight':
            # Add flight search suggestions
            if self.data_manager.data_loaded:
                airports = self.data_manager.get_available_airports()
                if airports:
                    response += f"\n\nAvailable airports include: {', '.join(airports[:5])}..."
        
        elif context and context.get('intent') == 'book_hotel':
            # Add hotel search suggestions
            if self.data_manager.data_loaded:
                locations = self.data_manager.get_available_locations()
                if locations:
                    response += f"\n\nAvailable locations include: {', '.join(locations[:5])}..."
        
        return response
    
    def process_message(self, message: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user message and return structured response
        
        This is the main method that orchestrates the entire NLP pipeline:
        1. Intent classification
        2. Entity extraction
        3. Response generation
        4. Context integration
        
        Args:
            message (str): User message to process
            user_context (Optional[Dict]): Previous conversation context
            
        Returns:
            Dict[str, Any]: Complete processing result including:
                - intent: Classified intent
                - confidence: Confidence score
                - entities: Extracted entities
                - response: Generated response
                - context: Updated context
        """
        try:
            # Step 1: Classify intent
            intent, confidence = self.classify_intent(message)
            
            # Step 2: Extract entities
            entities = self.extract_entities(message)
            
            # Step 3: Prepare context for response generation
            context = {
                'intent': intent,
                'entities': entities,
                'confidence': confidence
            }
            
            # Merge with user context if provided
            if user_context:
                context.update(user_context)
            
            # Step 4: Generate response
            response = self.generate_response(message, context)
            
            # Step 5: Prepare final result
            result = {
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'response': response,
                'context': context
            }
            
            logger.info(f"Processed message - Intent: {intent}, Confidence: {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'intent': 'error',
                'confidence': 0.0,
                'entities': {},
                'response': "I'm sorry, I encountered an error processing your message.",
                'context': {}
            }
    
    def get_intent_suggestions(self, partial_message: str) -> List[Tuple[str, float]]:
        """
        Get intent suggestions for partial user input
        
        This method provides real-time intent suggestions as the user types,
        useful for autocomplete or suggestion features.
        
        Args:
            partial_message (str): Partial user message
            
        Returns:
            List[Tuple[str, float]]: List of (intent, confidence) pairs
        """
        try:
            if len(partial_message.strip()) < 3:
                return []
            
            # Get all intent scores
            intent_scores = {}
            message_embedding = self.sentence_encoder.encode(partial_message.lower())
            
            for intent, patterns in self.intent_patterns.items():
                pattern_embeddings = self.sentence_encoder.encode(patterns)
                similarities = torch.cosine_similarity(
                    torch.tensor(message_embedding).unsqueeze(0),
                    torch.tensor(pattern_embeddings)
                )
                intent_scores[intent] = float(torch.max(similarities))
            
            # Sort by confidence and return top suggestions
            suggestions = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            return suggestions[:3]  # Return top 3 suggestions
            
        except Exception as e:
            logger.error(f"Error getting intent suggestions: {e}")
            return []
    
    def validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance extracted entities
        
        This method validates extracted entities against available data
        and enhances them with additional information when possible.
        
        Args:
            entities (Dict[str, Any]): Raw extracted entities
            
        Returns:
            Dict[str, Any]: Validated and enhanced entities
        """
        validated_entities = entities.copy()
        
        try:
            # Validate locations against available airports/cities
            if 'locations' in entities and self.data_manager.data_loaded:
                available_airports = self.data_manager.get_available_airports()
                available_locations = self.data_manager.get_available_locations()
                
                validated_locations = []
                for location in entities['locations']:
                    # Check if location matches available airports or cities
                    if any(location.lower() in airport.lower() for airport in available_airports):
                        validated_locations.append(location)
                    elif any(location.lower() in loc.lower() for loc in available_locations):
                        validated_locations.append(location)
                    else:
                        # Keep location but mark as unverified
                        validated_locations.append(f"{location} (unverified)")
                
                validated_entities['locations'] = validated_locations
            
            # Validate booking references
            if 'booking_reference' in entities:
                # Add validation logic for booking references
                ref = entities['booking_reference']
                if not re.match(r'^[A-Z]{2,3}\d{6,10}$', ref):
                    validated_entities['booking_reference'] = f"{ref} (invalid format)"
            
            logger.debug(f"Validated entities: {validated_entities}")
            
        except Exception as e:
            logger.error(f"Error validating entities: {e}")
        
        return validated_entities

# Global AI engine instance
ai_engine = AIEngine() 