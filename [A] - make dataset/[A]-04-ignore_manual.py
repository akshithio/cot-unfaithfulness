"""
[A]-04 ignore_manual.py:

Phase A: Data Collection
Part 04: Collate Final Dataset (w/o HITL)

--

This script creates a final dataset using ONLY LLM classifications where
the original classifier and verification classifier agree. It excludes:
1. All disagreements (where original ≠ verification)
2. All cases where both classifiers agree the result is "UNCLEAR"

The resulting dataset contains only high-confidence classifications where
both LLM runs agreed on either FAITHFUL or SELF-CORRECTED.

Input:
    - [A]-01-cot_results.jsonl (original classifications)
    - [A]-02-verification_report.jsonl (verification results)

Output:
    - [A]-04-auto_dataset.jsonl (automatically classified, high-confidence dataset)
    - [A].jsonl (canonical dataset file - same content as above)

Remarks:
    The final [A].jsonl dataset is generated as a result of running this script.
"""

import json
import os
from collections import defaultdict

class C:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

ORIGINAL_FILE = "[A]-01-cot_results.jsonl"
VERIFICATION_FILE = "[A]-02-verification_report.jsonl"
OUTPUT_FILE = "[A]-04-auto_dataset.jsonl"
CANONICAL_FILE = "[A].jsonl"

def load_jsonl(filepath):
    """Load JSONL file into list of dictionaries"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def classification_to_str(is_faithful):
    """Convert classification to readable string"""
    if is_faithful is True:
        return "FAITHFUL"
    elif is_faithful is False:
        return "SELF-CORRECTED"
    else:
        return "UNCLEAR"

def str_to_classification(text):
    """Convert string to classification value"""
    text = text.strip().upper()
    if text == "FAITHFUL":
        return True
    elif text == "SELF-CORRECTED":
        return False
    elif text == "UNCLEAR":
        return None
    return None

def main():
    print(f"{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Automatic Dataset Collation (Agreement-Only){C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    # Check input files
    if not os.path.exists(ORIGINAL_FILE):
        print(f"{C.FAIL}Error: {ORIGINAL_FILE} not found!{C.ENDC}")
        return
    if not os.path.exists(VERIFICATION_FILE):
        print(f"{C.FAIL}Error: {VERIFICATION_FILE} not found!{C.ENDC}")
        return
    
    # Check and remove existing output files
    if os.path.exists(OUTPUT_FILE):
        print(f"{C.WARNING}Removing existing output file: {OUTPUT_FILE}{C.ENDC}")
        os.remove(OUTPUT_FILE)
    
    if os.path.exists(CANONICAL_FILE):
        print(f"{C.WARNING}Removing existing canonical file: {CANONICAL_FILE}{C.ENDC}")
        os.remove(CANONICAL_FILE)
    
    print()
    
    # Load data
    print("Loading data files...")
    original_results = load_jsonl(ORIGINAL_FILE)
    verification_results = load_jsonl(VERIFICATION_FILE)
    
    print(f"  Original results: {len(original_results)} problems")
    print(f"  Verification results: {len(verification_results)} problems\n")
    
    # Create lookup dictionaries
    original_dict = {r['problem_id']: r for r in original_results}
    verification_dict = {r['problem_id']: r for r in verification_results}
    
    # Statistics tracking
    stats = {
        'total': len(original_results),
        'with_verification': 0,
        'agreements': 0,
        'disagreements': 0,
        'agreed_unclear': 0,
        'agreed_faithful': 0,
        'agreed_corrected': 0,
        'final_included': 0
    }
    
    final_dataset = []
    excluded_reasons = defaultdict(int)
    
    print(f"{C.OKGREEN}Processing classifications...{C.ENDC}\n")
    
    for original in original_results:
        problem_id = original['problem_id']
        
        # Get verification result
        verification = verification_dict.get(problem_id)
        if not verification:
            excluded_reasons['no_verification'] += 1
            continue
        
        stats['with_verification'] += 1
        
        # Check for agreement
        if not verification['agreement']:
            stats['disagreements'] += 1
            excluded_reasons['disagreement'] += 1
            continue
        
        stats['agreements'] += 1
        
        # Get classifications
        original_class = original['is_faithful']
        verification_class = str_to_classification(verification['verification_classification'])
        
        # Double-check they actually agree (should always be true if agreement=True)
        if original_class != verification_class:
            print(f"{C.WARNING}Warning: Problem {problem_id} marked as agreement but classifications differ!{C.ENDC}")
            excluded_reasons['agreement_mismatch'] += 1
            continue
        
        # Exclude if both agree it's UNCLEAR
        if original_class is None:
            stats['agreed_unclear'] += 1
            excluded_reasons['agreed_unclear'] += 1
            continue
        
        # Count agreement types
        if original_class is True:
            stats['agreed_faithful'] += 1
        else:  # original_class is False
            stats['agreed_corrected'] += 1
        
        # Include in final dataset
        final_entry = {
            # Problem info
            'problem_id': problem_id,
            'problem_text': original['problem_text'],
            'ground_truth_answer': original['ground_truth_answer'],
            
            # Original generation
            'original_cot': original['original_cot'],
            'original_answer': original['original_answer'],
            
            # Injection details
            'injection_point': original['injection_point'],
            'original_value': original['original_value'],
            'injected_value': original['injected_value'],
            
            # Continued reasoning
            'continued_cot': original['continued_cot'],
            'final_answer': original['final_answer'],
            
            # Classification (agreed upon by both LLM runs)
            'classification': original_class,
            'classification_str': classification_to_str(original_class),
            
            # Supporting evidence
            'original_notes': original['notes'],
            'verification_notes': verification['verification_notes'],
            
            # Metadata
            'classification_agreement': True,
            'data_source': 'llm_agreement'
        }
        
        final_dataset.append(final_entry)
        stats['final_included'] += 1
    
    # Save to both output files
    print(f"Saving dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Saving dataset to {CANONICAL_FILE}...")
    with open(CANONICAL_FILE, 'w') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + "\n")
    
    # Print statistics
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Dataset Statistics{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    print(f"{C.BOLD}Input:{C.ENDC}")
    print(f"  Total problems: {stats['total']}")
    print(f"  With verification: {stats['with_verification']}")
    
    print(f"\n{C.BOLD}Agreement Analysis:{C.ENDC}")
    print(f"  {C.OKGREEN}Agreements:{C.ENDC} {stats['agreements']} ({stats['agreements']/stats['with_verification']*100:.1f}%)")
    print(f"  {C.FAIL}Disagreements:{C.ENDC} {stats['disagreements']} ({stats['disagreements']/stats['with_verification']*100:.1f}%)")
    
    print(f"\n{C.BOLD}Agreement Breakdown:{C.ENDC}")
    print(f"  {C.FAIL}Agreed Faithful:{C.ENDC} {stats['agreed_faithful']}")
    print(f"  {C.OKGREEN}Agreed Self-Corrected:{C.ENDC} {stats['agreed_corrected']}")
    print(f"  {C.WARNING}Agreed Unclear:{C.ENDC} {stats['agreed_unclear']}")
    
    print(f"\n{C.BOLD}Exclusion Reasons:{C.ENDC}")
    for reason, count in sorted(excluded_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Final Dataset{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    print(f"{C.BOLD}Total Included:{C.ENDC} {stats['final_included']} problems")
    print(f"  {C.FAIL}Faithful (Propagated Error):{C.ENDC} {stats['agreed_faithful']} ({stats['agreed_faithful']/stats['final_included']*100:.1f}%)")
    print(f"  {C.OKGREEN}Self-Corrected:{C.ENDC} {stats['agreed_corrected']} ({stats['agreed_corrected']/stats['final_included']*100:.1f}%)")
    
    retention_rate = (stats['final_included'] / stats['total']) * 100
    print(f"\n{C.BOLD}Retention Rate:{C.ENDC} {retention_rate:.1f}% of original dataset")
    
    print(f"\n{C.OKGREEN}✓ High-confidence dataset saved to:{C.ENDC}")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {CANONICAL_FILE}")
    
    # Quality assessment
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Quality Assessment{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    if retention_rate >= 70:
        print(f"{C.OKGREEN}✓ Excellent: High classifier agreement (≥70% retention){C.ENDC}")
        print(f"  Dataset is ready for analysis")
    elif retention_rate >= 50:
        print(f"{C.OKGREEN}✓ Good: Reasonable classifier agreement (≥50% retention){C.ENDC}")
        print(f"  Dataset is usable but consider reviewing excluded cases")
    elif retention_rate >= 30:
        print(f"{C.WARNING}⚠ Fair: Moderate classifier agreement (30-50% retention){C.ENDC}")
        print(f"  Consider manual review of disagreements for higher quality")
    else:
        print(f"{C.FAIL}✗ Poor: Low classifier agreement (<30% retention){C.ENDC}")
        print(f"  Manual review strongly recommended")
        print(f"  Consider adjusting classifier prompt or using different model")
    
    # Balance check
    faithful_ratio = stats['agreed_faithful'] / stats['final_included'] if stats['final_included'] > 0 else 0
    if 0.3 <= faithful_ratio <= 0.7:
        print(f"\n{C.OKGREEN}✓ Dataset is balanced between faithful and self-corrected examples{C.ENDC}")
    elif faithful_ratio < 0.3:
        print(f"\n{C.WARNING}⚠ Dataset is skewed toward self-correction ({stats['agreed_corrected']}/{stats['final_included']}){C.ENDC}")
    else:
        print(f"\n{C.WARNING}⚠ Dataset is skewed toward faithful propagation ({stats['agreed_faithful']}/{stats['final_included']}){C.ENDC}")

if __name__ == "__main__":
    main()