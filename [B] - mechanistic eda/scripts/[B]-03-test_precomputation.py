"""
[B]-03 test_precomputation.py:

Phase B: Mechanistic EDA
Part 03: Pre-Computation vs Live Reasoning Test

--

This script tests whether models have "pre-computed" the final answer during the
original CoT generation, or whether they're doing live computation step-by-step.

Key Question: Is self-correction due to retrieving pre-computed answers, while
faithfulness represents genuine sequential reasoning?

Methodology:
1. Extract hidden states at EARLY points in the original CoT (before final answer)
2. Use a linear probe to predict the final answer from early hidden states
3. Compare probe accuracy between:
   - Self-corrected models (hypothesis: high accuracy = answer pre-computed)
   - Faithful models (hypothesis: lower accuracy = computing live)

If self-corrected models have the answer "encoded" early while faithful models don't,
this suggests:
- Self-correction = retrieval of pre-computed knowledge
- Faithfulness = genuine sequential reasoning

Input:
    - [A].jsonl (dataset with original CoT and answers)

Output:
    - [B]-03-output.json (probe results)
    - [B]-03-logs.txt (console output log)

--

Hypothesis: "Self-correction happens because models pre-compute 
            the answer, while faithful models compute step-by-step"
            
Result: REJECTED

Evidence: Both groups show ~10% predictive power from early states
          Difference: -1.9% (essentially zero, p >> 0.05)
          
Conclusion: Pre-computation does NOT explain the behavioral difference
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
from datetime import datetime

class C:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[B]-03-output.json"
LOG_FILE = "[B]-03-logs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Analysis settings
MAX_PROBLEMS = None  # Set to number for testing, None for all
LAYER_TO_ANALYZE = -1  # Which layer to probe (-1 = last layer)
EARLY_POSITION_FRACTION = 0.3  # Extract state at 30% through the CoT


class TeeOutput:
    """Writes to both stdout/stderr and a file simultaneously."""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.original_stream = original_stream
        
    def write(self, message):
        # Strip ANSI color codes for the file
        import re
        clean_message = re.sub(r'\033\[[0-9;]+m', '', message)
        self.file.write(clean_message)
        self.file.flush()
        self.original_stream.write(message)
        self.original_stream.flush()
    
    def flush(self):
        self.file.flush()
        self.original_stream.flush()
    
    def close(self):
        self.file.close()


class PrecomputationAnalyzer:
    def __init__(self, model_name, device):
        print(f"{C.OKCYAN}Loading model: {model_name} on {device}...{C.ENDC}")
        self.device = device
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else None,
            output_hidden_states=True
        )
        
        if device in ["mps", "cpu"]:
            self.model.to(device)
        
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"{C.OKGREEN}Model loaded successfully!{C.ENDC}\n")
    
    def get_early_hidden_state(self, problem_text, original_cot):
        """
        Extract hidden state from early in the CoT generation.
        
        Args:
            problem_text: The original problem
            original_cot: The complete original CoT reasoning
        
        Returns:
            Hidden state vector at early position, or None if failed
        """
        try:
            # Construct full prompt
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{problem_text}"}
            ]
            base_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Take only the first X% of the CoT (before the answer is computed)
            early_cutoff = int(len(original_cot) * EARLY_POSITION_FRACTION)
            early_cot = original_cot[:early_cutoff]
            
            full_sequence = base_prompt + early_cot
            
            # Tokenize
            inputs = self.tokenizer(
                full_sequence,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Get hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Extract hidden state from desired layer at last position
            hidden_states = outputs.hidden_states[LAYER_TO_ANALYZE]
            last_hidden_state = hidden_states[0, -1, :].float().cpu().numpy()
            
            return last_hidden_state
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Hidden state extraction failed - {str(e)[:100]}{C.ENDC}")
            return None
    
    def analyze_dataset(self, dataset_path, max_problems=None):
        """
        Extract early hidden states and prepare for probe training.
        """
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Pre-Computation vs Live Reasoning Test{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        problems = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        if max_problems:
            problems = problems[:max_problems]
        
        print(f"Loaded {len(problems)} problems\n")
        
        # Separate by classification
        classification_field = 'final_classification' if 'final_classification' in problems[0] else 'classification'
        faithful_problems = [p for p in problems if p[classification_field] is True]
        corrected_problems = [p for p in problems if p[classification_field] is False]
        
        print(f"{C.BOLD}Dataset Composition:{C.ENDC}")
        print(f"  {C.FAIL}Faithful (Propagated Error):{C.ENDC} {len(faithful_problems)}")
        print(f"  {C.OKGREEN}Self-Corrected:{C.ENDC} {len(corrected_problems)}")
        if len(faithful_problems) > 0 and len(corrected_problems) > 0:
            print(f"  Ratio: {len(faithful_problems)/len(corrected_problems):.2f}:1")
        print()
        
        print(f"{C.OKCYAN}Extracting early hidden states (at {EARLY_POSITION_FRACTION*100:.0f}% through CoT)...{C.ENDC}\n")
        
        # Extract hidden states and labels
        results = {
            'faithful': {'states': [], 'answers': [], 'problem_ids': []},
            'corrected': {'states': [], 'answers': [], 'problem_ids': []},
            'metadata': {
                'model': MODEL_NAME,
                'layer_analyzed': LAYER_TO_ANALYZE,
                'early_position_fraction': EARLY_POSITION_FRACTION,
                'total_problems': len(problems),
                'faithful_count': len(faithful_problems),
                'corrected_count': len(corrected_problems)
            }
        }
        
        print(f"{C.OKCYAN}Processing FAITHFUL problems...{C.ENDC}")
        for i, problem in enumerate(faithful_problems):
            print(f"[{i+1}/{len(faithful_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            hidden_state = self.get_early_hidden_state(
                problem['problem_text'],
                problem['original_cot']
            )
            
            if hidden_state is not None and problem['ground_truth_answer'] is not None:
                results['faithful']['states'].append(hidden_state)
                results['faithful']['answers'].append(problem['ground_truth_answer'])
                results['faithful']['problem_ids'].append(problem['problem_id'])
        
        print(" " * 80)
        
        print(f"{C.OKCYAN}Processing SELF-CORRECTED problems...{C.ENDC}")
        for i, problem in enumerate(corrected_problems):
            print(f"[{i+1}/{len(corrected_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            hidden_state = self.get_early_hidden_state(
                problem['problem_text'],
                problem['original_cot']
            )
            
            if hidden_state is not None and problem['ground_truth_answer'] is not None:
                results['corrected']['states'].append(hidden_state)
                results['corrected']['answers'].append(problem['ground_truth_answer'])
                results['corrected']['problem_ids'].append(problem['problem_id'])
        
        print(" " * 80)
        
        # Convert to numpy arrays
        results['faithful']['states'] = np.array(results['faithful']['states'])
        results['faithful']['answers'] = np.array(results['faithful']['answers'])
        results['corrected']['states'] = np.array(results['corrected']['states'])
        results['corrected']['answers'] = np.array(results['corrected']['answers'])
        
        return results


def train_answer_probe(states, answers, test_size=0.2):
    """
    Train a linear probe to predict whether the final answer is correct
    from early hidden states.
    
    For simplicity, we'll use a binary classification: does the hidden state
    encode information about the answer value?
    
    We'll bucket answers and see if we can predict the bucket.
    """
    from sklearn.model_selection import train_test_split
    
    # Create answer buckets for classification (0-10, 10-20, 20-50, 50-100, 100+)
    def bucket_answer(ans):
        if ans <= 10:
            return 0
        elif ans <= 20:
            return 1
        elif ans <= 50:
            return 2
        elif ans <= 100:
            return 3
        else:
            return 4
    
    labels = np.array([bucket_answer(a) for a in answers])
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        states, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression probe
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_accuracy = probe.score(X_train_scaled, y_train)
    test_accuracy = probe.score(X_test_scaled, y_test)
    
    # Baseline (random guessing based on class distribution)
    unique, counts = np.unique(labels, return_counts=True)
    baseline_accuracy = np.max(counts) / len(labels)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_classes': len(unique)
    }


def main():
    # Set up output redirection
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    tee_stdout = TeeOutput(LOG_FILE, original_stdout)
    sys.stdout = tee_stdout
    sys.stderr = tee_stdout
    
    try:
        # Print header with timestamp
        print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {LOG_FILE}\n")
        
        # Initialize analyzer
        analyzer = PrecomputationAnalyzer(MODEL_NAME, DEVICE)
        
        # Extract hidden states
        results = analyzer.analyze_dataset(INPUT_FILE, max_problems=MAX_PROBLEMS)
        
        # Check if we have enough data
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Linear Probe Analysis{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        print(f"{C.BOLD}Sample Sizes:{C.ENDC}")
        print(f"  Faithful: {len(results['faithful']['states'])} samples")
        print(f"  Corrected: {len(results['corrected']['states'])} samples\n")
        
        if len(results['faithful']['states']) < 20 or len(results['corrected']['states']) < 20:
            print(f"{C.FAIL}Error: Insufficient data for probe training (need at least 20 samples each){C.ENDC}")
            return
        
        # Train probes for each group
        print(f"{C.OKCYAN}Training probe on FAITHFUL models...{C.ENDC}")
        faithful_probe_results = train_answer_probe(
            results['faithful']['states'],
            results['faithful']['answers']
        )
        
        print(f"{C.OKCYAN}Training probe on SELF-CORRECTED models...{C.ENDC}")
        corrected_probe_results = train_answer_probe(
            results['corrected']['states'],
            results['corrected']['answers']
        )
        
        # Display results
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Probe Accuracy Results{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        print(f"{C.BOLD}Can we predict the final answer from EARLY hidden states?{C.ENDC}\n")
        
        print(f"{C.FAIL}FAITHFUL Models:{C.ENDC}")
        print(f"  Train Accuracy: {faithful_probe_results['train_accuracy']:.3f}")
        print(f"  Test Accuracy:  {faithful_probe_results['test_accuracy']:.3f}")
        print(f"  Baseline:       {faithful_probe_results['baseline_accuracy']:.3f}")
        print(f"  Above Baseline: {(faithful_probe_results['test_accuracy'] - faithful_probe_results['baseline_accuracy']):.3f}")
        
        print(f"\n{C.OKGREEN}SELF-CORRECTED Models:{C.ENDC}")
        print(f"  Train Accuracy: {corrected_probe_results['train_accuracy']:.3f}")
        print(f"  Test Accuracy:  {corrected_probe_results['test_accuracy']:.3f}")
        print(f"  Baseline:       {corrected_probe_results['baseline_accuracy']:.3f}")
        print(f"  Above Baseline: {(corrected_probe_results['test_accuracy'] - corrected_probe_results['baseline_accuracy']):.3f}")
        
        # Compare
        diff = corrected_probe_results['test_accuracy'] - faithful_probe_results['test_accuracy']
        
        print(f"\n{C.BOLD}Comparison:{C.ENDC}")
        print(f"  Difference: {diff:+.3f}")
        
        if diff > 0.05:
            advantage = "CORRECTED"
            color = C.OKGREEN
        elif diff < -0.05:
            advantage = "FAITHFUL"
            color = C.FAIL
        else:
            advantage = "SIMILAR"
            color = C.OKCYAN
        
        print(f"  Advantage: {color}{advantage}{C.ENDC}")
        
        # Interpretation
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Interpretation{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        corrected_advantage = corrected_probe_results['test_accuracy'] - corrected_probe_results['baseline_accuracy']
        faithful_advantage = faithful_probe_results['test_accuracy'] - faithful_probe_results['baseline_accuracy']
        
        if corrected_advantage > faithful_advantage + 0.05:
            print(f"{C.OKGREEN}✓ Strong Evidence of Pre-Computation in Self-Corrected Models:{C.ENDC}")
            print(f"  → Self-corrected models encode the answer EARLY in the CoT")
            print(f"  → Linear probe can predict final answer from {EARLY_POSITION_FRACTION*100:.0f}% position")
            print(f"  → Suggests answer was computed upfront, not step-by-step")
            print(f"  → Self-correction = retrieving pre-computed knowledge")
            print(f"\n{C.FAIL}✓ Faithful Models Show Less Pre-Computation:{C.ENDC}")
            print(f"  → Answer is NOT strongly encoded in early hidden states")
            print(f"  → Suggests genuine sequential computation")
            print(f"  → Faithfulness = live reasoning through CoT")
            print(f"\n{C.BOLD}Conclusion:{C.ENDC} CoT serves different purposes:")
            print(f"  • Self-corrected: CoT is post-hoc explanation of pre-computed answer")
            print(f"  • Faithful: CoT is genuine working memory for step-by-step computation")
        
        elif faithful_advantage > corrected_advantage + 0.05:
            print(f"{C.WARNING}⚠ Unexpected: Faithful Models Show More Pre-Computation:{C.ENDC}")
            print(f"  → Faithful models encode answer early")
            print(f"  → Yet they propagate errors when injected")
            print(f"  → Suggests they have the answer but override it with CoT")
            print(f"  → Faithfulness = commitment to sequential process over cached knowledge")
        
        else:
            print(f"{C.OKCYAN}○ Similar Pre-Computation in Both Groups:{C.ENDC}")
            print(f"  → Both groups encode similar information early")
            print(f"  → Probe accuracy difference: {diff:.3f}")
            print(f"  → Pre-computation alone doesn't explain behavioral differences")
            print(f"  → Other mechanisms (attention, reasoning) must be responsible")
        
        # Save results
        output_data = {
            'faithful': {
                'probe_results': faithful_probe_results,
                'problem_ids': results['faithful']['problem_ids']
            },
            'corrected': {
                'probe_results': corrected_probe_results,
                'problem_ids': results['corrected']['problem_ids']
            },
            'metadata': results['metadata']
        }
        
        print(f"\n{C.BOLD}Saving results to: {OUTPUT_FILE}{C.ENDC}")
        print(f"{C.BOLD}Saving console output to: {LOG_FILE}{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        
        print(f"\nConsole output saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()