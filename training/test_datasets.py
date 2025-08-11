#!/usr/bin/env python3
"""
Test SNIPS and MultiWOZ Datasets
================================

This script tests and analyzes SNIPS and MultiWOZ datasets
to see which one works best for travel bot training.

Author: Travel Bot Team
Date: 2024
"""

from datasets import load_dataset
import json

def test_snips_dataset():
    """Test and analyze SNIPS dataset"""
    print("=" * 60)
    print("🔍 TESTING SNIPS DATASET")
    print("=" * 60)
    
    try:
        # Load SNIPS dataset
        print("📥 Loading SNIPS dataset...")
        dataset = load_dataset("snips_built_in_intents")
        
        print(f"✅ SNIPS dataset loaded successfully!")
        print(f"📊 Available splits: {list(dataset.keys())}")
        
        # Analyze each split
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"\n📈 {split_name.upper()} split:")
            print(f"   Samples: {len(split_data):,}")
            
            if len(split_data) > 0:
                # Show sample structure
                sample = split_data[0]
                print(f"   Sample keys: {list(sample.keys())}")
                print(f"   Sample data: {sample}")
                
                # Analyze intents if available
                if 'intent' in sample:
                    intents = set(split_data['intent'])
                    print(f"   Intents: {sorted(intents)}")
                elif 'label' in sample:
                    labels = set(split_data['label'])
                    print(f"   Labels: {sorted(labels)}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading SNIPS: {e}")
        return None

def test_multiwoz_dataset():
    """Test and analyze MultiWOZ dataset"""
    print("\n" + "=" * 60)
    print("🔍 TESTING MULTIWOZ DATASET")
    print("=" * 60)
    
    try:
        # Load MultiWOZ dataset
        print("📥 Loading MultiWOZ dataset...")
        dataset = load_dataset("multiwoz_v2.1")
        
        print(f"✅ MultiWOZ dataset loaded successfully!")
        print(f"📊 Available splits: {list(dataset.keys())}")
        
        # Analyze each split
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"\n📈 {split_name.upper()} split:")
            print(f"   Conversations: {len(split_data):,}")
            
            if len(split_data) > 0:
                # Show sample structure
                sample = split_data[0]
                print(f"   Sample keys: {list(sample.keys())}")
                
                # Show conversation structure
                if 'turns' in sample:
                    print(f"   Number of turns: {len(sample['turns'])}")
                    if len(sample['turns']) > 0:
                        first_turn = sample['turns'][0]
                        print(f"   First turn keys: {list(first_turn.keys())}")
                        print(f"   First turn data: {first_turn}")
                
                # Show domains
                if 'domains' in sample:
                    print(f"   Domains: {sample['domains']}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading MultiWOZ: {e}")
        return None

def analyze_snips_for_travel():
    """Analyze SNIPS dataset for travel bot suitability"""
    print("\n" + "=" * 60)
    print("🎯 SNIPS ANALYSIS FOR TRAVEL BOT")
    print("=" * 60)
    
    try:
        dataset = load_dataset("snips_built_in_intents")
        
        # Get all intents
        all_intents = set(dataset['train']['intent'])
        print(f"📋 All SNIPS intents: {sorted(all_intents)}")
        
        # Analyze each intent
        print(f"\n📊 Intent Analysis:")
        for intent in sorted(all_intents):
            # Count samples for this intent
            count = sum(1 for item in dataset['train'] if item['intent'] == intent)
            print(f"   {intent}: {count} samples")
            
            # Show sample queries
            samples = [item for item in dataset['train'] if item['intent'] == intent][:2]
            for i, sample in enumerate(samples):
                print(f"     Example {i+1}: '{sample['text']}'")
        
        # Travel relevance analysis
        travel_relevant = {
            'BookRestaurant': 'High - Similar to hotel booking',
            'GetWeather': 'High - Direct travel relevance',
            'SearchCreativeWork': 'Medium - Could be adapted for travel search',
            'SearchScreeningEvent': 'Medium - Could be adapted for events',
            'AddToPlaylist': 'Low - Not travel related',
            'PlayMusic': 'Low - Not travel related',
            'RateBook': 'Low - Not travel related'
        }
        
        print(f"\n🎯 Travel Relevance Analysis:")
        for intent, relevance in travel_relevant.items():
            print(f"   {intent}: {relevance}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error analyzing SNIPS: {e}")
        return None

def analyze_multiwoz_for_travel():
    """Analyze MultiWOZ dataset for travel bot suitability"""
    print("\n" + "=" * 60)
    print("🎯 MULTIWOZ ANALYSIS FOR TRAVEL BOT")
    print("=" * 60)
    
    try:
        dataset = load_dataset("multiwoz_v2.1")
        
        # Analyze domains
        all_domains = set()
        for conversation in dataset['train']:
            all_domains.update(conversation['domains'])
        
        print(f"📋 All MultiWOZ domains: {sorted(all_domains)}")
        
        # Travel-relevant domains
        travel_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'bus']
        print(f"\n🎯 Travel-relevant domains: {travel_domains}")
        
        # Count conversations with travel domains
        travel_conversations = 0
        for conversation in dataset['train']:
            if any(domain in conversation['domains'] for domain in travel_domains):
                travel_conversations += 1
        
        print(f"📊 Travel conversations: {travel_conversations:,} out of {len(dataset['train']):,}")
        
        # Show sample travel conversation
        print(f"\n📋 Sample travel conversation:")
        for conversation in dataset['train']:
            if any(domain in conversation['domains'] for domain in travel_domains):
                print(f"   Domains: {conversation['domains']}")
                print(f"   Turns: {len(conversation['turns'])}")
                if len(conversation['turns']) > 0:
                    first_turn = conversation['turns'][0]
                    print(f"   First turn: '{first_turn.get('utterance', 'N/A')}'")
                break
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error analyzing MultiWOZ: {e}")
        return None

def create_travel_training_data():
    """Create travel-specific training data from available datasets"""
    print("\n" + "=" * 60)
    print("🚀 CREATING TRAVEL TRAINING DATA")
    print("=" * 60)
    
    travel_data = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    # Try to extract travel-relevant data from SNIPS
    try:
        snips_dataset = load_dataset("snips_built_in_intents")
        
        # Map SNIPS intents to travel intents
        intent_mapping = {
            'BookRestaurant': 'book_hotel',
            'GetWeather': 'weather_info',
            'SearchCreativeWork': 'search_travel',
            'SearchScreeningEvent': 'search_events'
        }
        
        print("📥 Extracting travel-relevant data from SNIPS...")
        
        for split_name in ['train', 'validation']:
            if split_name in snips_dataset:
                for item in snips_dataset[split_name]:
                    if item['intent'] in intent_mapping:
                        travel_data[split_name].append({
                            'text': item['text'],
                            'intent': intent_mapping[item['intent']]
                        })
        
        print(f"✅ Extracted {len(travel_data['train'])} training samples from SNIPS")
        
    except Exception as e:
        print(f"⚠️ Could not extract from SNIPS: {e}")
    
    # Add synthetic travel data
    synthetic_data = [
        # Flight booking
        {'text': 'I want to book a flight from New York to London', 'intent': 'book_flight'},
        {'text': 'Book me a flight to Paris', 'intent': 'book_flight'},
        {'text': 'I need a flight from Boston to Miami', 'intent': 'book_flight'},
        {'text': 'Show me flights to Tokyo', 'intent': 'book_flight'},
        {'text': 'Find flights from Los Angeles to Chicago', 'intent': 'book_flight'},
        
        # Hotel booking
        {'text': 'Book me a hotel in Paris', 'intent': 'book_hotel'},
        {'text': 'I need a hotel room in London', 'intent': 'book_hotel'},
        {'text': 'Find hotels in Tokyo', 'intent': 'book_hotel'},
        {'text': 'Book a hotel in New York', 'intent': 'book_hotel'},
        {'text': 'I want to stay at a hotel in Miami', 'intent': 'book_hotel'},
        
        # Travel updates
        {'text': 'What time does my flight depart', 'intent': 'travel_updates'},
        {'text': 'Is my flight on time', 'intent': 'travel_updates'},
        {'text': 'Check flight status', 'intent': 'travel_updates'},
        {'text': 'What gate is my flight', 'intent': 'travel_updates'},
        {'text': 'Is my flight delayed', 'intent': 'travel_updates'},
        
        # Weather info
        {'text': 'What is the weather in Paris', 'intent': 'weather_info'},
        {'text': 'Weather forecast for London', 'intent': 'weather_info'},
        {'text': 'How is the weather in Tokyo', 'intent': 'weather_info'},
        {'text': 'Weather in New York', 'intent': 'weather_info'},
        {'text': 'What is the temperature in Miami', 'intent': 'weather_info'},
        
        # Booking management
        {'text': 'Cancel my reservation', 'intent': 'cancel_booking'},
        {'text': 'I want to cancel my flight', 'intent': 'cancel_booking'},
        {'text': 'Cancel hotel booking', 'intent': 'cancel_booking'},
        {'text': 'I need to cancel my trip', 'intent': 'cancel_booking'},
        {'text': 'Cancel my hotel room', 'intent': 'cancel_booking'},
        
        # Check booking
        {'text': 'Check my booking status', 'intent': 'check_booking'},
        {'text': 'What is my booking reference', 'intent': 'check_booking'},
        {'text': 'Show me my reservation', 'intent': 'check_booking'},
        {'text': 'Check my flight booking', 'intent': 'check_booking'},
        {'text': 'What is my hotel booking', 'intent': 'check_booking'},
        
        # Modify booking
        {'text': 'I want to change my flight', 'intent': 'modify_booking'},
        {'text': 'Modify my hotel booking', 'intent': 'modify_booking'},
        {'text': 'Change my reservation', 'intent': 'modify_booking'},
        {'text': 'I need to modify my booking', 'intent': 'modify_booking'},
        {'text': 'Change flight dates', 'intent': 'modify_booking'},
        
        # General help
        {'text': 'Help me with booking', 'intent': 'general_help'},
        {'text': 'What can you do', 'intent': 'general_help'},
        {'text': 'How do I book a flight', 'intent': 'general_help'},
        {'text': 'I need help', 'intent': 'general_help'},
        {'text': 'What services do you offer', 'intent': 'general_help'},
    ]
    
    # Add synthetic data to training
    travel_data['train'].extend(synthetic_data[:40])  # Add most to training
    travel_data['validation'].extend(synthetic_data[40:45])  # Add some to validation
    travel_data['test'].extend(synthetic_data[45:])  # Add rest to test
    
    print(f"✅ Added {len(synthetic_data)} synthetic travel samples")
    
    # Show final dataset stats
    print(f"\n📊 Final Travel Dataset:")
    print(f"   Train: {len(travel_data['train'])} samples")
    print(f"   Validation: {len(travel_data['validation'])} samples")
    print(f"   Test: {len(travel_data['test'])} samples")
    
    # Show intents
    all_intents = set()
    for split_data in travel_data.values():
        for item in split_data:
            all_intents.add(item['intent'])
    
    print(f"   Intents: {sorted(all_intents)}")
    
    return travel_data

def main():
    """Main function to test datasets"""
    print("🚀 SNIPS and MultiWOZ Dataset Testing")
    print("=" * 60)
    
    # Test SNIPS
    snips_dataset = test_snips_dataset()
    snips_analysis = analyze_snips_for_travel()
    
    # Test MultiWOZ
    multiwoz_dataset = test_multiwoz_dataset()
    multiwoz_analysis = analyze_multiwoz_for_travel()
    
    # Create travel training data
    travel_data = create_travel_training_data()
    
    print(f"\n🎉 Testing completed!")
    print(f"📊 Results:")
    print(f"   SNIPS: {'✅ Available' if snips_dataset else '❌ Not available'}")
    print(f"   MultiWOZ: {'✅ Available' if multiwoz_dataset else '❌ Not available'}")
    print(f"   Travel Data: ✅ Created with {len(travel_data['train'])} samples")

if __name__ == "__main__":
    main() 