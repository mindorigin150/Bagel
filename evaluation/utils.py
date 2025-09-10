import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union

def get_setting_from_id(item_id: str) -> str:
    """
    Extract setting type from item ID.
    
    Args:
        item_id: The item identifier
        
    Returns:
        Setting type ('around', 'rotation', 'translation', 'among', 'other')
    """
    if not item_id:
        return 'other'
    
    item_id_lower = item_id.lower()
    
    if 'around' in item_id_lower:
        return 'around'
    elif 'rotation' in item_id_lower:
        return 'rotation'
    elif 'translation' in item_id_lower:
        return 'translation'
    elif 'among' in item_id_lower:
        return 'among'
    else:
        return 'other'



def initialize_basic_results_structure() -> Dict:
    """
    Initialize the basic results data structure.
    
    Returns:
        Empty results structure with all required fields
    """
    # Basic settings to track - maintain consistent order with original
    settings = ['around', 'rotation', 'translation', 'among', 'other']
    
    # Define which settings should be included in overall metrics
    # Translation is excluded from overall metrics but still tracked
    settings_to_include = {
        'around': True, 
        'rotation': True, 
        'translation': False,  # Exclude translation from overall metrics
        'among': True, 
        'other': True
    }
    
    # Initialize results
    results = {
        'total': 0,
        'unfiltered_total': 0,  # Original total (all settings)
        'gen_cogmap_correct': 0,
        'gen_cogmap_accuracy': 0.0,
        'settings': {setting: {
            'total': 0, 
            'gen_cogmap_correct': 0,
            'gen_cogmap_accuracy': 0.0,
            'include_in_overall': settings_to_include.get(setting, True),  # Flag for filtering
        } for setting in settings}
    }
    
    return results


def update_accuracy_metrics(results: Dict) -> Dict:
    """
    Update accuracy metrics for all settings.
    
    Args:
        results: Results dictionary to update
        
    Returns:
        Updated results dictionary
    """
    # Calculate setting-specific accuracy
    for setting, stats in results['settings'].items():
        if stats['total'] > 0:
            stats['gen_cogmap_accuracy'] = stats['gen_cogmap_correct'] / stats['total']
        else:
            stats['gen_cogmap_accuracy'] = 0.0
    
    # Calculate overall accuracy using filtered total
    if results['total'] > 0:
        results['gen_cogmap_accuracy'] = results['gen_cogmap_correct'] / results['total']
    else:
        results['gen_cogmap_accuracy'] = 0.0
    
    return results




"""
I/O utilities for evaluation data loading and result saving.

This module provides functions to:
1. Load data from JSONL files
2. Save evaluation results
3. Print results in a readable format
"""
import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional


def load_jsonl_data(jsonl_path: str) -> List[Dict]:
    """
    Load data from a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of loaded JSON objects
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_json_results(results: Dict, output_path: str) -> None:
    """
    Save evaluation results as JSON.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save the results to
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")


def print_basic_results(results: Dict) -> None:
    """
    Print basic evaluation results in a readable format.
    
    Args:
        results: Evaluation results dictionary
    """
    total = results.get('total', 0)
    unfiltered_total = results.get('unfiltered_total', total)
    correct = results.get('gen_cogmap_correct', 0)
    accuracy = results.get('gen_cogmap_accuracy', 0.0)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Total examples: {unfiltered_total} (Evaluated: {total}, excluding translation)")
    print(f"Answer accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    
    # Print results by setting
    print(f"\n=== RESULTS BY SETTING ===")
    for setting, stats in results.get('settings', {}).items():
        setting_total = stats.get('total', 0)
        setting_correct = stats.get('gen_cogmap_correct', 0) 
        setting_accuracy = stats.get('gen_cogmap_accuracy', 0.0)
        include_in_overall = stats.get('include_in_overall', True)
        
        status = "" if include_in_overall else " (excluded from overall)"
        print(f"{setting.capitalize()}: {setting_accuracy*100:.2f}% ({setting_correct}/{setting_total}){status}")
