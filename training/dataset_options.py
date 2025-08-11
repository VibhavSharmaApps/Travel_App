#!/usr/bin/env python3
"""
Famous Datasets for Intent Classification
========================================

This script explores and tests various famous datasets available on Hugging Face
for training intent classification models.

Author: Travel Bot Team
Date: 2024
"""

from datasets import load_dataset
import json

def test_dataset_availability():
    """Test availability of famous intent classification datasets"""
    
    datasets_to_test = [
        # Most Popular Intent Classification Datasets
        {
            'name': 'snips_built_in_intents',
            'description': 'SNIPS NLU Benchmark - Most popular intent dataset',
            'size': '~2K samples, 7 intents',
            'domain': 'General NLU'
        },
        {
            'name': 'clinc_oos',
            'description': 'CLINC Out-of-Scope Intent Detection',
            'size': '~23K samples, 150 intents',
            'domain': 'Multi-domain'
        },
        {
            'name': 'banking77',
            'description': 'Banking Customer Service Intents',
            'size': '~13K samples, 77 intents',
            'domain': 'Banking'
        },
        {
            'name': 'hwu64',
            'description': 'HWU64 Multi-domain Intent Dataset',
            'size': '~11K samples, 64 intents',
            'domain': 'Multi-domain'
        },
        {
            'name': 'mcid',
            'description': 'Microsoft Conversational Intelligence Dataset',
            'size': '~46K samples, 66 intents',
            'domain': 'Customer Service'
        },
        {
            'name': 'rostd',
            'description': 'Robust Spoken Text Detection',
            'size': '~8K samples, 3 intents',
            'domain': 'Spoken Language'
        },
        {
            'name': 'trec',
            'description': 'TREC Question Classification',
            'size': '~6K samples, 6 intents',
            'domain': 'Question Classification'
        },
        {
            'name': 'ag_news',
            'description': 'AG News Classification',
            'size': '~127K samples, 4 intents',
            'domain': 'News Classification'
        },
        {
            'name': 'dbpedia_14',
            'description': 'DBpedia Entity Classification',
            'size': '~560K samples, 14 intents',
            'domain': 'Entity Classification'
        },
        {
            'name': 'yahoo_answers_topics',
            'description': 'Yahoo Answers Topic Classification',
            'size': '~1.4M samples, 10 intents',
            'domain': 'Topic Classification'
        }
    ]
    
    print("🔍 Testing Famous Intent Classification Datasets")
    print("=" * 60)
    
    available_datasets = []
    
    for dataset_info in datasets_to_test:
        try:
            print(f"\n📊 Testing: {dataset_info['name']}")
            print(f"   Description: {dataset_info['description']}")
            print(f"   Size: {dataset_info['size']}")
            print(f"   Domain: {dataset_info['domain']}")
            
            # Try to load the dataset
            dataset = load_dataset(dataset_info['name'])
            
            # Get basic info
            train_size = len(dataset['train']) if 'train' in dataset else 0
            test_size = len(dataset['test']) if 'test' in dataset else 0
            validation_size = len(dataset['validation']) if 'validation' in dataset else 0
            
            print(f"   ✅ Available!")
            print(f"   📈 Train: {train_size:,} samples")
            print(f"   📈 Test: {test_size:,} samples")
            print(f"   📈 Validation: {validation_size:,} samples")
            
            # Show sample data
            if train_size > 0:
                sample = dataset['train'][0]
                print(f"   📋 Sample: {str(sample)[:100]}...")
            
            available_datasets.append({
                **dataset_info,
                'dataset': dataset,
                'train_size': train_size,
                'test_size': test_size,
                'validation_size': validation_size
            })
            
        except Exception as e:
            print(f"   ❌ Not available: {str(e)[:50]}...")
    
    return available_datasets

def analyze_snips_dataset():
    """Detailed analysis of SNIPS dataset"""
    print("\n" + "="*60)
    print("📊 DETAILED ANALYSIS: SNIPS Dataset")
    print("="*60)
    
    try:
        # Load SNIPS dataset
        dataset = load_dataset("snips_built_in_intents")
        
        print(f"✅ SNIPS dataset loaded successfully!")
        print(f"📈 Train samples: {len(dataset['train']):,}")
        print(f"📈 Test samples: {len(dataset['test']):,}")
        print(f"📈 Validation samples: {len(dataset['validation']):,}")
        
        # Analyze intents
        train_intents = dataset['train']['intent']
        unique_intents = set(train_intents)
        
        print(f"\n🎯 Intents found: {len(unique_intents)}")
        for intent in sorted(unique_intents):
            count = train_intents.count(intent)
            print(f"   - {intent}: {count} samples")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        for i in range(3):
            sample = dataset['train'][i]
            print(f"   {i+1}. Text: '{sample['text']}'")
            print(f"      Intent: {sample['intent']}")
            print()
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading SNIPS: {e}")
        return None

def analyze_clinc_dataset():
    """Detailed analysis of CLINC dataset"""
    print("\n" + "="*60)
    print("📊 DETAILED ANALYSIS: CLINC Dataset")
    print("="*60)
    
    try:
        # Load CLINC dataset
        dataset = load_dataset("clinc_oos")
        
        print(f"✅ CLINC dataset loaded successfully!")
        print(f"📈 Train samples: {len(dataset['train']):,}")
        print(f"📈 Test samples: {len(dataset['test']):,}")
        print(f"📈 Validation samples: {len(dataset['validation']):,}")
        
        # Analyze intents
        train_intents = dataset['train']['intent']
        unique_intents = set(train_intents)
        
        print(f"\n🎯 Intents found: {len(unique_intents)}")
        print("   (Showing first 20 intents)")
        for intent in sorted(list(unique_intents))[:20]:
            count = train_intents.count(intent)
            print(f"   - {intent}: {count} samples")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        for i in range(3):
            sample = dataset['train'][i]
            print(f"   {i+1}. Text: '{sample['text']}'")
            print(f"      Intent: {sample['intent']}")
            print()
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading CLINC: {e}")
        return None

def analyze_banking_dataset():
    """Detailed analysis of Banking77 dataset"""
    print("\n" + "="*60)
    print("📊 DETAILED ANALYSIS: Banking77 Dataset")
    print("="*60)
    
    try:
        # Load Banking77 dataset
        dataset = load_dataset("banking77")
        
        print(f"✅ Banking77 dataset loaded successfully!")
        print(f"📈 Train samples: {len(dataset['train']):,}")
        print(f"📈 Test samples: {len(dataset['test']):,}")
        
        # Analyze intents
        train_intents = dataset['train']['label']
        unique_intents = set(train_intents)
        
        print(f"\n🎯 Intents found: {len(unique_intents)}")
        print("   (Showing first 15 intents)")
        for intent in sorted(list(unique_intents))[:15]:
            count = train_intents.count(intent)
            print(f"   - Intent {intent}: {count} samples")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        for i in range(3):
            sample = dataset['train'][i]
            print(f"   {i+1}. Text: '{sample['text']}'")
            print(f"      Intent: {sample['label']}")
            print()
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading Banking77: {e}")
        return None

def recommend_dataset():
    """Recommend the best dataset for travel bot"""
    print("\n" + "="*60)
    print("🎯 RECOMMENDATION FOR TRAVEL BOT")
    print("="*60)
    
    recommendations = [
        {
            'name': 'snips_built_in_intents',
            'reason': 'Most popular and reliable intent dataset',
            'pros': ['Well-maintained', 'Good documentation', 'Balanced classes'],
            'cons': ['Not travel-specific', 'Only 7 intents']
        },
        {
            'name': 'clinc_oos',
            'reason': 'Large multi-domain dataset with 150 intents',
            'pros': ['Large dataset', 'Many intents', 'Good for generalization'],
            'cons': ['Complex', 'Not travel-specific']
        },
        {
            'name': 'banking77',
            'reason': 'Customer service intents (similar to travel booking)',
            'pros': ['Customer service domain', '77 intents', 'Good quality'],
            'cons': ['Banking-specific', 'Not travel-focused']
        }
    ]
    
    print("🏆 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Pros: {', '.join(rec['pros'])}")
        print(f"   Cons: {', '.join(rec['cons'])}")
    
    print(f"\n💡 MY RECOMMENDATION:")
    print(f"   Start with SNIPS dataset for quick training and testing.")
    print(f"   Then try CLINC for more comprehensive training.")
    print(f"   Finally, create a custom travel dataset combining these.")

def main():
    """Main function to test and analyze datasets"""
    print("🚀 Famous Intent Classification Datasets Explorer")
    print("=" * 60)
    
    # Test all datasets
    available_datasets = test_dataset_availability()
    
    # Detailed analysis of top datasets
    snips_dataset = analyze_snips_dataset()
    clinc_dataset = analyze_clinc_dataset()
    banking_dataset = analyze_banking_dataset()
    
    # Recommendations
    recommend_dataset()
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Found {len(available_datasets)} available datasets")
    
    return available_datasets

if __name__ == "__main__":
    main() 