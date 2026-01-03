"""
[B]-02 analyze_embeddings.py:

Phase B: Mechanistic EDA
Part 02: Internal Belief Analysis via Embedding Geometry

--

This script investigates whether models internally "believe" the errors they propagate
by analyzing the geometric relationship between hidden states and value embeddings.

Key Question: When a model outputs an incorrect value, does its internal representation
align with the error (believing it) or the truth (knowing better)?

Methodology:
1. Extract hidden states at the injection point (after processing the error)
2. Get embeddings for both injected (wrong) and original (correct) values
3. Calculate cosine similarity: hidden_state ↔ error_embedding vs truth_embedding
4. Compute "belief bias": sim(error) - sim(truth)
   - Positive bias = internal state aligned with error (truly believes it)
   - Negative bias = internal state aligned with truth (knows better)
5. Compare between FAITHFUL vs SELF-CORRECTED groups

Expected Results:
- If FAITHFUL models show positive bias → they genuinely believe the error
- If FAITHFUL models show negative bias → they know the truth but propagate anyway
  (evidence of thought-action decoupling)

Input:
    - [A].jsonl (classified problems with injections)

Output:
    - [B]-02-output.json (belief bias scores per problem)
    - [B]-02-logs.txt (console output log)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from scipy import stats
import sys
import re
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

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[B]-02-output.json"
LOG_FILE = "[B]-02-logs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Analysis settings
MAX_PROBLEMS = None  # Set to number for testing, None for all
LAYER_TO_ANALYZE = -1  # Which layer's hidden states to use (-1 = last layer)

class TeeOutput:
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.original_stream = original_stream
        
    def write(self, message):
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

class EmbeddingAnalyzer:
    def __init__(self, model_name, device):
        print(f"{C.OKCYAN}Loading model: {model_name} on {device}...{C.ENDC}")
        self.device = device
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else None,
            output_hidden_states=True
        )
        
        if device in ["mps", "cpu"]:
            self.model.to(device)
        
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get embedding layer
        self.embedding_layer = self.model.get_input_embeddings()
        
        print(f"{C.OKGREEN}Model loaded successfully!{C.ENDC}")
        print(f"{C.OKGREEN}Embedding dimension: {self.embedding_layer.embedding_dim}{C.ENDC}\n")
    
    def get_value_embedding(self, value_str):
        """
        Get the embedding vector for a numerical value.
        Returns averaged embedding if value spans multiple tokens.
        """
        # Tokenize the value
        tokens = self.tokenizer.encode(str(value_str), add_special_tokens=False)
        token_ids = torch.tensor([tokens]).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.embedding_layer(token_ids)  # (1, num_tokens, embed_dim)
        
        # Average if multiple tokens
        avg_embedding = embeddings.mean(dim=1).squeeze(0)  # (embed_dim,)
        
        return avg_embedding
    
    def find_injection_position(self, full_text, original_value, injected_value, injection_context):
        """
        Find the token position where the injection occurred.
        Returns the position after the injected value.
        """
        # Try to find using context
        context_stripped = injection_context.strip()
        if context_stripped in full_text:
            context_pos = full_text.find(context_stripped)
            # Look for injected value near this context
            search_start = max(0, context_pos - 30)
            search_end = min(len(full_text), context_pos + len(injection_context) + 30)
            local_search = full_text[search_start:search_end]
            
            if str(injected_value) in local_search:
                # Found it, return position after injected value
                return full_text[:search_end].find(str(injected_value)) + len(str(injected_value))
        
        # Fallback: search for pattern "= injected_value"
        pattern = f"= {injected_value}"
        pos = full_text.find(pattern)
        if pos != -1:
            return pos + len(pattern)
        
        # Last resort: just find injected value
        pos = full_text.find(str(injected_value))
        if pos != -1:
            return pos + len(str(injected_value))
        
        return None
    
    def get_hidden_state_at_position(self, problem_text, full_continuation, char_position):
        """
        Get the hidden state at a specific character position in the text.
        
        Args:
            problem_text: Original problem
            full_continuation: The continued reasoning (includes injection)
            char_position: Character position where we want the hidden state
        
        Returns:
            Hidden state vector (tensor) or None if failed
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
            
            # Full sequence up to our position of interest
            full_sequence = base_prompt + full_continuation[:char_position]
            
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
            # outputs.hidden_states is a tuple of (num_layers,) each (batch, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states[LAYER_TO_ANALYZE]  # (1, seq_len, hidden_dim)
            last_hidden_state = hidden_states[0, -1, :]  # (hidden_dim,)
            
            return last_hidden_state
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Hidden state extraction failed - {str(e)[:100]}{C.ENDC}")
            return None
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0), 
            vec2.unsqueeze(0)
        ).item()
    
    def analyze_belief_bias(self, problem):
        """
        Analyze whether the model's internal state aligns with error or truth.
        
        Returns:
            {
                'problem_id': int,
                'belief_bias': float,  # sim(error) - sim(truth)
                'sim_to_error': float,
                'sim_to_truth': float,
                'success': bool
            }
        """
        # Get value embeddings
        error_embedding = self.get_value_embedding(problem['injected_value'])
        truth_embedding = self.get_value_embedding(problem['original_value'])
        
        # Find injection position in character space
        char_position = self.find_injection_position(
            problem['continued_cot'],
            problem['original_value'],
            problem['injected_value'],
            problem['injection_point']
        )
        
        if char_position is None:
            return {
                'problem_id': problem['problem_id'],
                'belief_bias': None,
                'sim_to_error': None,
                'sim_to_truth': None,
                'success': False,
                'error': 'Could not find injection position'
            }
        
        # Get hidden state at that position
        hidden_state = self.get_hidden_state_at_position(
            problem['problem_text'],
            problem['continued_cot'],
            char_position
        )
        
        if hidden_state is None:
            return {
                'problem_id': problem['problem_id'],
                'belief_bias': None,
                'sim_to_error': None,
                'sim_to_truth': None,
                'success': False,
                'error': 'Could not extract hidden state'
            }
        
        # Calculate similarities
        sim_to_error = self.cosine_similarity(hidden_state, error_embedding)
        sim_to_truth = self.cosine_similarity(hidden_state, truth_embedding)
        
        # Belief bias: positive = aligned with error, negative = aligned with truth
        belief_bias = sim_to_error - sim_to_truth
        
        return {
            'problem_id': problem['problem_id'],
            'belief_bias': belief_bias,
            'sim_to_error': sim_to_error,
            'sim_to_truth': sim_to_truth,
            'success': True,
            'injected_value': problem['injected_value'],
            'original_value': problem['original_value']
        }
    
    def analyze_dataset(self, dataset_path, max_problems=None):
        """
        Analyze belief bias across the entire dataset.
        """
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Internal Belief Analysis via Embedding Geometry{C.ENDC}")
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
        
        # Separate by classification (try both field names for compatibility)
        classification_field = 'final_classification' if 'final_classification' in problems[0] else 'classification'
        faithful_problems = [p for p in problems if p[classification_field] is True]
        corrected_problems = [p for p in problems if p[classification_field] is False]
        
        print(f"{C.BOLD}Dataset Composition:{C.ENDC}")
        print(f"  {C.FAIL}Faithful (Propagated Error):{C.ENDC} {len(faithful_problems)}")
        print(f"  {C.OKGREEN}Self-Corrected:{C.ENDC} {len(corrected_problems)}")
        if len(faithful_problems) > 0 and len(corrected_problems) > 0:
            print(f"  Ratio: {len(faithful_problems)/len(corrected_problems):.2f}:1")
        print()
        
        # Analyze each group
        results = {
            'faithful': [],
            'corrected': [],
            'metadata': {
                'model': MODEL_NAME,
                'layer_analyzed': LAYER_TO_ANALYZE,
                'embedding_dim': self.embedding_layer.embedding_dim,
                'total_problems': len(problems),
                'faithful_count': len(faithful_problems),
                'corrected_count': len(corrected_problems)
            }
        }
        
        print(f"{C.OKCYAN}Analyzing FAITHFUL problems...{C.ENDC}")
        for i, problem in enumerate(faithful_problems):
            print(f"[{i+1}/{len(faithful_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            result = self.analyze_belief_bias(problem)
            results['faithful'].append(result)
        
        print(" " * 80)
        
        print(f"{C.OKCYAN}Analyzing SELF-CORRECTED problems...{C.ENDC}")
        for i, problem in enumerate(corrected_problems):
            print(f"[{i+1}/{len(corrected_problems)}] Problem {problem['problem_id']}...", end="\r")
            
            result = self.analyze_belief_bias(problem)
            results['corrected'].append(result)
        
        print(" " * 80)
        
        return results


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
        analyzer = EmbeddingAnalyzer(MODEL_NAME, DEVICE)
        
        # Run analysis
        results = analyzer.analyze_dataset(INPUT_FILE, max_problems=MAX_PROBLEMS)
        
        # Save results
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Compute statistics
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Statistical Analysis{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        # Extract successful analyses
        faithful_biases = [r['belief_bias'] for r in results['faithful'] if r['success']]
        corrected_biases = [r['belief_bias'] for r in results['corrected'] if r['success']]
        
        faithful_sim_error = [r['sim_to_error'] for r in results['faithful'] if r['success']]
        faithful_sim_truth = [r['sim_to_truth'] for r in results['faithful'] if r['success']]
        corrected_sim_error = [r['sim_to_error'] for r in results['corrected'] if r['success']]
        corrected_sim_truth = [r['sim_to_truth'] for r in results['corrected'] if r['success']]
        
        if not faithful_biases or not corrected_biases:
            print(f"{C.FAIL}Error: Insufficient data for analysis{C.ENDC}")
            print(f"  Faithful samples: {len(faithful_biases)}")
            print(f"  Corrected samples: {len(corrected_biases)}")
            return
        
        print(f"{C.BOLD}Sample Sizes:{C.ENDC}")
        print(f"  Faithful: {len(faithful_biases)}/{len(results['faithful'])} successful")
        print(f"  Corrected: {len(corrected_biases)}/{len(results['corrected'])} successful\n")
        
        # Belief Bias Statistics
        faithful_bias_mean = np.mean(faithful_biases)
        faithful_bias_std = np.std(faithful_biases)
        faithful_bias_se = faithful_bias_std / np.sqrt(len(faithful_biases))
        
        corrected_bias_mean = np.mean(corrected_biases)
        corrected_bias_std = np.std(corrected_biases)
        corrected_bias_se = corrected_bias_std / np.sqrt(len(corrected_biases))
        
        print(f"{C.BOLD}Belief Bias (sim_error - sim_truth):{C.ENDC}\n")
        
        print(f"{C.FAIL}FAITHFUL (Propagated Error):{C.ENDC}")
        print(f"  Mean: {faithful_bias_mean:+.6f} ± {faithful_bias_se:.6f}")
        print(f"  Std:  {faithful_bias_std:.6f}")
        print(f"  Interpretation: {'Aligned with ERROR' if faithful_bias_mean > 0 else 'Aligned with TRUTH'}")
        
        print(f"\n{C.OKGREEN}SELF-CORRECTED:{C.ENDC}")
        print(f"  Mean: {corrected_bias_mean:+.6f} ± {corrected_bias_se:.6f}")
        print(f"  Std:  {corrected_bias_std:.6f}")
        print(f"  Interpretation: {'Aligned with ERROR' if corrected_bias_mean > 0 else 'Aligned with TRUTH'}")
        
        # Detailed similarity breakdown
        print(f"\n{C.BOLD}Detailed Similarity Breakdown:{C.ENDC}\n")
        
        print(f"{C.FAIL}FAITHFUL:{C.ENDC}")
        print(f"  Similarity to Error: {np.mean(faithful_sim_error):.6f} ± {np.std(faithful_sim_error)/np.sqrt(len(faithful_sim_error)):.6f}")
        print(f"  Similarity to Truth: {np.mean(faithful_sim_truth):.6f} ± {np.std(faithful_sim_truth)/np.sqrt(len(faithful_sim_truth)):.6f}")
        
        print(f"\n{C.OKGREEN}SELF-CORRECTED:{C.ENDC}")
        print(f"  Similarity to Error: {np.mean(corrected_sim_error):.6f} ± {np.std(corrected_sim_error)/np.sqrt(len(corrected_sim_error)):.6f}")
        print(f"  Similarity to Truth: {np.mean(corrected_sim_truth):.6f} ± {np.std(corrected_sim_truth)/np.sqrt(len(corrected_sim_truth)):.6f}")
        
        # Statistical comparison
        diff = faithful_bias_mean - corrected_bias_mean
        
        # Cohen's d
        pooled_std = np.sqrt(((len(faithful_biases)-1)*faithful_bias_std**2 + 
                              (len(corrected_biases)-1)*corrected_bias_std**2) / 
                             (len(faithful_biases) + len(corrected_biases) - 2))
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        # t-test
        t_stat, p_value = stats.ttest_ind(faithful_biases, corrected_biases)
        
        print(f"\n{C.BOLD}Statistical Comparison:{C.ENDC}")
        print(f"  Difference: {diff:+.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        # Significance
        if p_value < 0.001:
            sig_str = f"{C.OKGREEN}*** (p < 0.001){C.ENDC}"
        elif p_value < 0.01:
            sig_str = f"{C.OKGREEN}** (p < 0.01){C.ENDC}"
        elif p_value < 0.05:
            sig_str = f"{C.OKGREEN}* (p < 0.05){C.ENDC}"
        else:
            sig_str = f"{C.WARNING}n.s. (p ≥ 0.05){C.ENDC}"
        
        print(f"  Significance: {sig_str}")
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_str = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_str = "small"
        elif abs(cohens_d) < 0.8:
            effect_str = "medium"
        else:
            effect_str = "large"
        
        print(f"  Effect size: {effect_str}")
        
        # Interpret results
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Interpretation{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        # Check for thought-action decoupling
        if faithful_bias_mean < 0 and corrected_bias_mean < 0:
            print(f"{C.OKGREEN}✓ Strong Evidence of Thought-Action Decoupling:{C.ENDC}")
            print(f"  → BOTH groups internally aligned with TRUTH (negative bias)")
            print(f"  → Faithful models KNOW the correct value but propagate error anyway")
            print(f"  → Internal belief differs from external output")
            print(f"  → Error propagation is a behavioral choice, not cognitive error")
        
        elif faithful_bias_mean > 0 and corrected_bias_mean < 0:
            print(f"{C.WARNING}⚠ Models Differ in Internal Beliefs:{C.ENDC}")
            print(f"  → Faithful models: internally aligned with ERROR (believe it)")
            print(f"  → Corrected models: internally aligned with TRUTH (know better)")
            print(f"  → Suggests faithful models genuinely accept the injected value")
            print(f"  → Error propagation reflects true internal representation")
        
        elif faithful_bias_mean > 0 and corrected_bias_mean > 0:
            print(f"{C.FAIL}✗ Both Groups Aligned with Error:{C.ENDC}")
            print(f"  → Both internally represent the error value")
            print(f"  → Corrected models overcome bad representation through reasoning")
            print(f"  → No evidence of thought-action decoupling")
        
        else:  # faithful negative, corrected positive
            print(f"{C.OKCYAN}○ Unexpected Pattern:{C.ENDC}")
            print(f"  → Faithful: aligned with truth, Corrected: aligned with error")
            print(f"  → Counterintuitive finding - warrants further investigation")
        
        # Additional significance check
        if p_value < 0.05:
            print(f"\n{C.OKGREEN}Statistical Significance Confirmed:{C.ENDC}")
            print(f"  → Difference is statistically significant (p = {p_value:.4f})")
            print(f"  → Effect size: {effect_str} (d = {cohens_d:.3f})")
        else:
            print(f"\n{C.WARNING}No Significant Difference:{C.ENDC}")
            print(f"  → Groups show similar internal representations (p = {p_value:.4f})")
            print(f"  → Behavioral differences not reflected in embedding geometry")
        
        print(f"\n{C.BOLD}Saved detailed results to: {OUTPUT_FILE}{C.ENDC}")
        print(f"{C.BOLD}Saved console output to: {LOG_FILE}{C.ENDC}")
        
        print(f"\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        
        print(f"\nConsole output saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()