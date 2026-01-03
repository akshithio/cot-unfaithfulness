"""
[B]-05 activation_patching.py

Phase B: Mechanistic EDA
Part 05: Activation Patching to Test Causal Role of Residual Stream

This script tests whether residual stream activations causally mediate error
propagation by:
1. Running model with error → get corrupted activations
2. Running model with correct value → get clean activations  
3. Patching clean activations into corrupted run at error position
4. Measuring if this "cures" the model (breaks error propagation)

Expected Result:
- If faithful models propagate errors via residual stream, patching should
  restore correct reasoning
- If self-corrected models ignore residual stream, patching should have no effect
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

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
OUTPUT_FILE = "[B]-05-output.json"
LOG_FILE = "[B]-05-logs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MAX_PROBLEMS = None  # None = all problems
LAYERS_TO_PATCH = None  # Will be auto-detected based on model architecture
GENERATION_LENGTH = 1024 
DEBUG = True  # Enable detailed debugging output

class Logger:
    """Logs output to both console and file, stripping color codes from file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        clean_message = re.sub(r'\033\[[0-9;]+m', '', message)
        self.log.write(clean_message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class ActivationPatcher:
    def __init__(self, model_name, device):
        print(f"{C.OKCYAN}Loading model: {model_name} on {device}...{C.ENDC}")
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else None,
            attn_implementation="eager",
            output_hidden_states=True
        )
        
        if device in ["mps", "cpu"]:
            self.model.to(device)
        
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Detect number of layers
        self.num_layers = len(self.model.model.layers)
        print(f"{C.BOLD}Model has {self.num_layers} layers{C.ENDC}")
        
        # Auto-select layers to patch (last 4 layers)
        global LAYERS_TO_PATCH
        if LAYERS_TO_PATCH is None:
            LAYERS_TO_PATCH = list(range(self.num_layers))
        print(f"{C.BOLD}Will patch layers: {LAYERS_TO_PATCH}{C.ENDC}")
        
        print(f"{C.OKGREEN}Model loaded successfully!{C.ENDC}\n")
    
    def extract_final_answer(self, text):
        """
        Extract numerical answer from text, supporting both decimals and fractions.
        COPIED FROM CAUSAL INTERVENTION SCRIPT - proven to work!
        """
        answer_patterns = [
            (r'\\boxed\{\s*\\frac\{(\d+)\}\{(\d+)\}\s*\}', 'frac'),
            (r'\\boxed\{\s*(\d+)\\frac\{(\d+)\}\{(\d+)\}\s*\}', 'mixed'),
            (r'\\boxed\{\s*(\$?-?[\d,]+(?:\.\d+)?).*?\}', 'decimal'),
            (r'\\frac\{(\d+)\}\{(\d+)\}', 'frac'),
            (r'(?:answer|result)(?:\s+is)?[:\s]+(\d+)/(\d+)', 'text_frac'),
            (r'(?:final answer|answer|result)(?:\s+is)?[:\s]+\$?\s*([\d,]+(?:\.\d+)?)', 'decimal'),
            (r'####\s*([\d,]+(?:\.\d+)?)', 'decimal'),
            (r'(?:^|\n)(?:the answer is|answer:)\s*\$?\s*([\d,]+(?:\.\d+)?)', 'decimal'),
            (r'=\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:\n|$)', 'decimal')
        ]
        
        for pattern, pattern_type in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    if pattern_type == 'frac':
                        numerator, denominator = matches[-1]
                        return float(numerator) / float(denominator)
                    elif pattern_type == 'mixed':
                        whole, numerator, denominator = matches[-1]
                        return float(whole) + (float(numerator) / float(denominator))
                    elif pattern_type == 'text_frac':
                        numerator, denominator = matches[-1]
                        return float(numerator) / float(denominator)
                    else:
                        return float(matches[-1].replace(',', '').replace('$', '').strip())
                except (ValueError, ZeroDivisionError): 
                    continue
        
        return None
    
    def answers_match(self, extracted, ground_truth):
        """
        Check if extracted answer matches ground truth with tolerance.
        Returns True if correct, False if incorrect, None if extraction failed.
        """
        if extracted is None:
            return None
        
        try:
            extracted_float = float(extracted)
            ground_truth_float = float(ground_truth)
            
            # Use relative tolerance: 1% of value or 0.01 absolute
            tolerance = 0.01 * abs(ground_truth_float) + 0.01
            is_correct = abs(extracted_float - ground_truth_float) < tolerance
            
            return is_correct
        except (ValueError, TypeError):
            return None
    
    def find_value_positions(self, text: str, value: str) -> List[int]:
        """Find token positions where value appears in text."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        value_tokens = self.tokenizer.encode(str(value), add_special_tokens=False)
        
        positions = []
        for i in range(len(tokens) - len(value_tokens) + 1):
            if tokens[i:i+len(value_tokens)] == value_tokens:
                positions.extend(range(i, i + len(value_tokens)))
        
        return positions
    
    def get_clean_and_corrupted_states(
        self, 
        problem_text: str,
        continuation_prefix: str,  # NOW EXPECTS PREFIX, NOT FULL CoT
        injected_value: str,
        original_value: str
    ) -> Optional[Dict]:
        """
        Get hidden states from both clean and corrupted forward passes.
        Returns positions and states needed for patching.
        
        CRITICAL: continuation_prefix should be the CoT up to shortly after the error,
        NOT the complete solution. This allows the model to continue solving.
        """
        try:
            # Build prompts
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{problem_text}"}
            ]
            base_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Corrupted version (with injected error in prefix)
            corrupted_full = base_prompt + continuation_prefix
            corrupted_inputs = self.tokenizer(
                corrupted_full,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Find error positions in prefix
            error_positions_in_cont = self.find_value_positions(
                continuation_prefix,
                injected_value
            )
            
            if not error_positions_in_cont:
                if DEBUG:
                    print(f"\n{C.WARNING}  Could not find injected value '{injected_value}' in continuation{C.ENDC}")
                return None
            
            # Adjust for base prompt
            base_len = len(self.tokenizer.encode(base_prompt, add_special_tokens=False))
            error_start = error_positions_in_cont[0] + base_len
            error_end = error_positions_in_cont[-1] + base_len + 1
            
            # Clean version (with correct value in prefix)
            continuation_clean = continuation_prefix.replace(
                str(injected_value),
                str(original_value),
                1  # Only first occurrence
            )
            
            clean_full = base_prompt + continuation_clean
            clean_inputs = self.tokenizer(
                clean_full,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Forward pass: Corrupted
            with torch.no_grad():
                corrupted_outputs = self.model(
                    input_ids=corrupted_inputs.input_ids,
                    attention_mask=corrupted_inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Forward pass: Clean
            with torch.no_grad():
                clean_outputs = self.model(
                    input_ids=clean_inputs.input_ids,
                    attention_mask=clean_inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            return {
                'corrupted_hidden_states': corrupted_outputs.hidden_states,
                'clean_hidden_states': clean_outputs.hidden_states,
                'error_start': error_start,
                'error_end': error_end,
                'corrupted_input_ids': corrupted_inputs.input_ids,
                'clean_input_ids': clean_inputs.input_ids,
                'corrupted_attention_mask': corrupted_inputs.attention_mask,
                'clean_attention_mask': clean_inputs.attention_mask,
                'base_prompt': base_prompt,
                'continuation_with_error': continuation_prefix,
                'continuation_clean': continuation_clean
            }
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Failed to get states - {str(e)[:150]}{C.ENDC}")
            import traceback
            if DEBUG:
                traceback.print_exc()
            return None
    
    def patch_and_generate(
        self,
        states_dict: Dict,
        layer_to_patch: int,
        ground_truth_answer: float
    ) -> Optional[Dict]:
        """
        Patch clean activations into corrupted run and generate continuation.
        
        Returns metrics about generation quality.
        """
        try:
            # Validate layer index
            if layer_to_patch >= self.num_layers:
                print(f"\n{C.WARNING}Skipping layer {layer_to_patch} (model has {self.num_layers} layers){C.ENDC}")
                return None
            
            error_start = states_dict['error_start']
            error_end = states_dict['error_end']
            
            # Create patched input by starting from corrupted sequence
            current_ids = states_dict['corrupted_input_ids'].clone()
            attention_mask = states_dict['corrupted_attention_mask'].clone()
            
            # Hook to patch activations during generation
            def create_hook(layer_idx):
                def hook_fn(module, input, output):
                    # output is tuple: (hidden_states, ...)
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    
                    # If we're at error position, patch with clean activations
                    seq_len = hidden_states.shape[1]
                    if error_start < seq_len:
                        patch_end = min(error_end, seq_len)
                        clean_states = states_dict['clean_hidden_states'][layer_idx]
                        
                        # Make sure we don't exceed clean_states bounds
                        if patch_end <= clean_states.shape[1]:
                            # Patch error positions
                            hidden_states[:, error_start:patch_end, :] = \
                                clean_states[:, error_start:patch_end, :]
                    
                    return (hidden_states,) if isinstance(output, tuple) else hidden_states
                
                return hook_fn
            
            # Register hooks for specified layer
            hooks = []
            layer = self.model.model.layers[layer_to_patch]
            hook = layer.register_forward_hook(create_hook(layer_to_patch))
            hooks.append(hook)
            
            # Generate with patching active
            with torch.no_grad():
                outputs = self.model.generate(
                    current_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=GENERATION_LENGTH,
                    do_sample=False,  # Greedy for reproducibility
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Decode generated text
            generated_ids = outputs[0][current_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract answer
            extracted_answer = self.extract_final_answer(generated_text)
            is_correct = self.answers_match(extracted_answer, ground_truth_answer)
            
            if DEBUG:
                print(f"\n{C.OKCYAN}    Layer {layer_to_patch} Patched Generation:{C.ENDC}")
                print(f"      Generated (first 150 chars): {generated_text[:150]}...")
                print(f"      Extracted answer: {extracted_answer}")
                print(f"      Ground truth: {ground_truth_answer}")
                print(f"      Is correct: {is_correct}")
            
            return {
                'generated_text': generated_text,
                'generated_length': len(generated_ids),
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'layer_patched': layer_to_patch
            }
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Patching failed - {str(e)[:150]}{C.ENDC}")
            import traceback
            if DEBUG:
                traceback.print_exc()
            return None
    
    def baseline_generate(
        self, 
        states_dict: Dict, 
        use_clean: bool, 
        ground_truth_answer: float
    ) -> Optional[Dict]:
        """Generate without patching (baseline comparison)."""
        try:
            input_ids = states_dict['clean_input_ids'] if use_clean else states_dict['corrupted_input_ids']
            attention_mask = states_dict['clean_attention_mask'] if use_clean else states_dict['corrupted_attention_mask']
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=GENERATION_LENGTH,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = outputs[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract answer
            extracted_answer = self.extract_final_answer(generated_text)
            is_correct = self.answers_match(extracted_answer, ground_truth_answer)
            
            condition = 'clean_baseline' if use_clean else 'corrupted_baseline'
            
            if DEBUG:
                print(f"\n{C.OKCYAN}  {condition.upper()}:{C.ENDC}")
                print(f"    Generated (first 150 chars): {generated_text[:150]}...")
                print(f"    Extracted answer: {extracted_answer}")
                print(f"    Ground truth: {ground_truth_answer}")
                print(f"    Is correct: {is_correct}")
            
            return {
                'generated_text': generated_text,
                'generated_length': len(generated_ids),
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'condition': condition
            }
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Baseline generation failed - {str(e)[:150]}{C.ENDC}")
            import traceback
            if DEBUG:
                traceback.print_exc()
            return None
    
    def analyze_problem(
        self,
        problem: Dict,
        verbose: bool = False
    ) -> Optional[Dict]:
        """
        Complete analysis for one problem:
        1. Get clean and corrupted states
        2. Generate baselines (no patching)
        3. Generate with patching at each layer
        4. Compare outcomes
        
        CRITICAL FIX: Create prefix from continued_cot (not use full CoT)
        """
        try:
            if verbose or DEBUG:
                print(f"\n{C.BOLD}{'='*80}{C.ENDC}")
                print(f"{C.BOLD}Analyzing problem {problem['problem_id']}:{C.ENDC}")
                print(f"  Injected: {problem['injected_value']}, Correct: {problem['original_value']}")
                print(f"  Ground truth answer: {problem.get('ground_truth_answer', 'N/A')}")
            
            # CREATE PREFIX (not full CoT) - CRITICAL FIX
            continued_cot = problem['continued_cot']
            injection_pos = continued_cot.find(str(problem['injected_value']))
            
            if injection_pos == -1:
                if DEBUG:
                    print(f"{C.WARNING}  Could not find injection position for problem {problem['problem_id']}{C.ENDC}")
                return None
            
            # Get prefix up to shortly after the error (just like causal intervention)
            prefix_end = min(injection_pos + len(str(problem['injected_value'])) + 100, 
                            len(continued_cot))
            continued_cot_prefix = continued_cot[:prefix_end]
            
            if DEBUG:
                print(f"\n{C.OKCYAN}  Using CoT prefix (length {len(continued_cot_prefix)} of {len(continued_cot)}):{C.ENDC}")
                print(f"    {continued_cot_prefix[:200]}...")
            
            # Get states using PREFIX (not full CoT)
            states = self.get_clean_and_corrupted_states(
                problem['problem_text'],
                continued_cot_prefix,  # USE PREFIX, not full CoT
                problem['injected_value'],
                problem['original_value']
            )
            
            if states is None:
                if DEBUG:
                    print(f"{C.FAIL}  Failed to get states for problem {problem['problem_id']}{C.ENDC}")
                return None
            
            # Add values to states for checking
            states['original_value'] = problem['original_value']
            states['injected_value'] = problem['injected_value']
            
            # Use ground_truth_answer from dataset
            ground_truth = problem.get('ground_truth_answer')
            if ground_truth is None:
                if DEBUG:
                    print(f"{C.WARNING}  No ground_truth_answer for problem {problem['problem_id']}{C.ENDC}")
                return None
            
            # Baselines
            corrupted_baseline = self.baseline_generate(states, use_clean=False, ground_truth_answer=ground_truth)
            clean_baseline = self.baseline_generate(states, use_clean=True, ground_truth_answer=ground_truth)
            
            if corrupted_baseline is None or clean_baseline is None:
                if DEBUG:
                    print(f"{C.FAIL}  Failed baselines for problem {problem['problem_id']}{C.ENDC}")
                return None
            
            # Patching experiments
            patching_results = []
            for layer in LAYERS_TO_PATCH:
                if not DEBUG:
                    print(f"  Patching layer {layer}...", end="\r")
                
                result = self.patch_and_generate(states, layer, ground_truth)
                if result is not None:
                    patching_results.append(result)
            
            if not DEBUG:
                print(" " * 50, end="\r")
            
            result = {
                'problem_id': problem['problem_id'],
                'classification': problem['classification'],
                'injected_value': problem['injected_value'],
                'original_value': problem['original_value'],
                'ground_truth_answer': ground_truth,
                'corrupted_baseline': corrupted_baseline,
                'clean_baseline': clean_baseline,
                'patching_results': patching_results
            }
            
            if verbose or DEBUG:
                print(f"\n{C.OKGREEN}  ✓ Problem {problem['problem_id']} complete{C.ENDC}")
            
            return result
            
        except Exception as e:
            print(f"\n{C.WARNING}Warning: Problem analysis failed - {str(e)[:150]}{C.ENDC}")
            import traceback
            if DEBUG:
                traceback.print_exc()
            return None
    
    def analyze_dataset(self, dataset_path: str, max_problems: Optional[int] = None):
        """Analyze multiple problems with activation patching."""
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Activation Patching Analysis{C.ENDC}")
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
        faithful = [p for p in problems if p['classification'] is True]
        corrected = [p for p in problems if p['classification'] is False]
        
        print(f"{C.BOLD}Dataset Composition:{C.ENDC}")
        print(f"  Faithful: {len(faithful)}")
        print(f"  Self-Corrected: {len(corrected)}")
        print()
        
        results = {
            'faithful': [],
            'corrected': [],
            'metadata': {
                'model': MODEL_NAME,
                'layers_patched': LAYERS_TO_PATCH,
                'generation_length': GENERATION_LENGTH,
                'total_problems': len(problems),
                'faithful_count': len(faithful),
                'corrected_count': len(corrected)
            }
        }
        
        # Analyze faithful
        print(f"{C.OKCYAN}Analyzing FAITHFUL problems...{C.ENDC}")
        for i, problem in enumerate(faithful):
            if not DEBUG:
                print(f"[{i+1}/{len(faithful)}] Problem {problem['problem_id']}...", end="\r")
            analysis = self.analyze_problem(problem, verbose=(i == 0 and not DEBUG))
            if analysis:
                results['faithful'].append(analysis)
        if not DEBUG:
            print(" " * 80)
        
        # Analyze corrected
        print(f"\n{C.OKCYAN}Analyzing SELF-CORRECTED problems...{C.ENDC}")
        for i, problem in enumerate(corrected):
            if not DEBUG:
                print(f"[{i+1}/{len(corrected)}] Problem {problem['problem_id']}...", end="\r")
            analysis = self.analyze_problem(problem, verbose=(i == 0 and not DEBUG))
            if analysis:
                results['corrected'].append(analysis)
        if not DEBUG:
            print(" " * 80)
        
        return results


def compute_patching_effectiveness(results: Dict):
    """Compute how effective patching was at recovering correct behavior."""
    
    def compute_recovery_rate(problems: List[Dict]) -> Dict:
        """Compute recovery metrics for a set of problems."""
        total = len(problems)
        if total == 0:
            return {}
        
        # Baseline metrics
        corrupted_correct = sum(1 for p in problems 
                               if p['corrupted_baseline']['is_correct'] is True)
        clean_correct = sum(1 for p in problems 
                           if p['clean_baseline']['is_correct'] is True)
        
        # Patching metrics per layer
        layer_recovery = {}
        for layer in LAYERS_TO_PATCH:
            layer_results = [p for p in problems 
                           if any(r['layer_patched'] == layer for r in p['patching_results'])]
            
            if layer_results:
                patched_correct = sum(
                    1 for p in layer_results
                    if any(r['layer_patched'] == layer and r['is_correct'] is True
                          for r in p['patching_results'])
                )
                
                layer_recovery[layer] = {
                    'recovery_rate': patched_correct / len(layer_results),
                    'n': len(layer_results)
                }
        
        return {
            'corrupted_baseline_accuracy': corrupted_correct / total,
            'clean_baseline_accuracy': clean_correct / total,
            'layer_recovery': layer_recovery,
            'total_problems': total
        }
    
    faithful_metrics = compute_recovery_rate(results['faithful'])
    corrected_metrics = compute_recovery_rate(results['corrected'])
    
    return {
        'faithful': faithful_metrics,
        'corrected': corrected_metrics
    }


def print_results_summary(results: Dict, effectiveness: Dict):
    """Pretty print summary of patching results."""
    print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
    print(f"{C.HEADER}Patching Effectiveness Summary{C.ENDC}")
    print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
    
    for category in ['faithful', 'corrected']:
        metrics = effectiveness[category]
        if not metrics:
            continue
            
        print(f"{C.BOLD}{category.upper()} Models:{C.ENDC}")
        print(f"  Corrupted Baseline: {metrics['corrupted_baseline_accuracy']:.2%}")
        print(f"  Clean Baseline: {metrics['clean_baseline_accuracy']:.2%}")
        
        if metrics['layer_recovery']:
            print(f"  Layer Recovery:")
            for layer, stats in sorted(metrics['layer_recovery'].items()):
                improvement = stats['recovery_rate'] - metrics['corrupted_baseline_accuracy']
                color = C.OKGREEN if improvement > 0.1 else C.WARNING if improvement > 0 else C.FAIL
                print(f"    Layer {layer}: {color}{stats['recovery_rate']:.2%}{C.ENDC} "
                      f"(Δ{improvement:+.2%}, n={stats['n']})")
        else:
            print(f"  {C.WARNING}No patching results available{C.ENDC}")
        print()


def main():
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    
    try:
        print(f"Activation Patching Analysis Log")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Initialize patcher
        patcher = ActivationPatcher(MODEL_NAME, DEVICE)
        
        # Run analysis
        results = patcher.analyze_dataset(INPUT_FILE, max_problems=MAX_PROBLEMS)
        
        # Compute effectiveness
        effectiveness = compute_patching_effectiveness(results)
        
        # Save results
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        output_data = {
            'results': results,
            'effectiveness': effectiveness
        }
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Print summary
        print_results_summary(results, effectiveness)
        
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.BOLD}Files Saved:{C.ENDC}")
        print(f"  • {OUTPUT_FILE}")
        print(f"  • {LOG_FILE}")
        print(f"\n{C.OKGREEN}Analysis complete!{C.ENDC}")
        
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n{C.OKGREEN}Full output saved to {LOG_FILE}{C.ENDC}")


if __name__ == "__main__":
    main()