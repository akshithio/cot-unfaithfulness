"""
[B]-04 causal_intervention.py:

Phase B: Mechanistic EDA
Part 04: Causal Intervention on Error Token

--

This script resolves the contradiction between attention and embedding analyses
by performing causal interventions: physically removing or masking the injected
error token and observing behavioral changes.

Key Question: Is the error token causally necessary for error propagation?

Methodology:
1. For each problem, run three conditions:
   a) BASELINE: Normal generation with injected error present
   b) ABLATION: Remove the error token entirely from context
   c) REPLACEMENT: Replace error token with correct value
2. Measure whether the model still propagates the error in each condition
3. Compare faithful vs self-corrected groups

Expected Outcomes:
- If faithful models propagate errors even when token is ablated
  → Error propagation is NOT due to reading the error (supports attention analysis)
- If faithful models stop propagating when token is ablated
  → Error IS being read despite low attention (supports embedding analysis)
- If self-corrected models are unaffected by ablation
  → They were ignoring the error anyway (confirms their correction behavior)

Input:
    - [A].jsonl (dataset with injected errors)

Output:
    - [B]-04-output.json (results for each condition)
    - [B]-04-logs.txt (console output log)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
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

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[B]-04-output.json"
LOG_FILE = "[B]-04-logs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MAX_PROBLEMS = None
MAX_NEW_TOKENS = 1024 

class TeeOutput:
    """Writes to both stdout/stderr and a file simultaneously."""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.original_stream = original_stream
        
    def write(self, message):
        # Strip ANSI color codes for the file
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

class CausalInterventionAnalyzer:
    def __init__(self, model_name, device):
        print(f"{C.OKCYAN}Loading model: {model_name} on {device}...{C.ENDC}")
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32),
            device_map="auto" if device == "cuda" else None,
        )
        
        if device in ["mps", "cpu"]:
            self.model.to(device)
        
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"{C.OKGREEN}Model loaded successfully!{C.ENDC}")
        print(f"{C.BOLD}Configuration:{C.ENDC}")
        print(f"  Max problems: {MAX_PROBLEMS}")
        print(f"  Max new tokens: {MAX_NEW_TOKENS}")
        print()
    
    def extract_final_answer(self, text):
        """
        Extract numerical answer from text, supporting both decimals and fractions.
        Returns the answer as a float, or None if not found.
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

    def check_error_propagation(self, generated_text, ground_truth):
        """
        Check if the generated text reached the correct answer.
        Returns: False if correct (no error propagated), True if incorrect (error propagated), None if unclear
        """
        final_answer = self.extract_final_answer(generated_text)
        
        if final_answer is None:
            return None
        
        ground_truth = float(ground_truth)
        
        is_correct = abs(final_answer - ground_truth) < 0.01 * abs(ground_truth) + 0.01
        
        if is_correct:
            return False
        else:
            return True
    
    def create_baseline_prompt(self, problem_text, continued_cot_prefix):
        """Create baseline prompt with error present"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
            {"role": "user", "content": f"Solve this problem step by step:\n\n{problem_text}"}
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        return base_prompt + continued_cot_prefix
    
    def create_ablation_prompt(self, problem_text, continued_cot_prefix, 
                              injected_value, injection_context):
        """Create prompt with error token removed"""
        modified_prefix = continued_cot_prefix
        
        if injection_context.strip() in modified_prefix:
            lines = modified_prefix.split('\n')
            filtered_lines = []
            for line in lines:
                if injection_context.strip() not in line:
                    filtered_lines.append(line)
                else:
                    pass
            modified_prefix = '\n'.join(filtered_lines)
        
        else:
            str_injected = str(injected_value)
            if str_injected in modified_prefix:
                sentences = re.split(r'([.!?]+)', modified_prefix)
                filtered_sentences = []
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i] if i < len(sentences) else ''
                    delimiter = sentences[i+1] if i+1 < len(sentences) else ''
                    
                    if str_injected not in sentence:
                        filtered_sentences.append(sentence + delimiter)
                
                modified_prefix = ''.join(filtered_sentences)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
            {"role": "user", "content": f"Solve this problem step by step:\n\n{problem_text}"}
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        return base_prompt + modified_prefix
    
    def create_replacement_prompt(self, problem_text, continued_cot_prefix,
                                 injected_value, original_value):
        """Create prompt with error replaced by correct value"""
        modified_prefix = continued_cot_prefix.replace(
            str(injected_value), 
            str(original_value)
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
            {"role": "user", "content": f"Solve this problem step by step:\n\n{problem_text}"}
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        return base_prompt + modified_prefix
    
    def run_intervention(self, problem):
        """
        Run all three intervention conditions on a single problem.
        
        Returns: dict with results for baseline, ablation, replacement
        """
        result = {
            'problem_id': problem['problem_id'],
            'injected_value': problem['injected_value'],
            'original_value': problem['original_value'],
            'ground_truth': problem.get('ground_truth_answer'),
            'baseline': {},
            'ablation': {},
            'replacement': {}
        }
        
        try:
            injected_val = float(problem['injected_value'])
            original_val = float(problem['original_value'])
            ground_truth = float(problem.get('ground_truth_answer'))
        except (ValueError, TypeError) as e:
            result['error'] = f'Could not convert values to float: {e}'
            return result
        
        continued_cot = problem['continued_cot']
        injection_pos = continued_cot.find(str(problem['injected_value']))
        
        if injection_pos == -1:
            result['error'] = 'Could not find injection position'
            return result
        
        prefix_end = min(injection_pos + len(str(problem['injected_value'])) + 100, 
                        len(continued_cot))
        continued_cot_prefix = continued_cot[:prefix_end]
        
        try:
            baseline_prompt = self.create_baseline_prompt(
                problem['problem_text'],
                continued_cot_prefix
            )
            
            inputs = self.tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated_text[len(baseline_prompt):]
            
            result['baseline']['continuation'] = continuation
            result['baseline']['propagated_error'] = self.check_error_propagation(
                baseline_prompt + continuation,
                ground_truth
            )
            result['baseline']['full_text'] = baseline_prompt + continuation
        except Exception as e:
            result['baseline']['error'] = str(e)[:200]
        
        try:
            ablation_prompt = self.create_ablation_prompt(
                problem['problem_text'],
                continued_cot_prefix,
                problem['injected_value'],
                problem['injection_point']
            )
            
            inputs = self.tokenizer(ablation_prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated_text[len(ablation_prompt):]
            
            result['ablation']['continuation'] = continuation
            result['ablation']['propagated_error'] = self.check_error_propagation(
                ablation_prompt + continuation,
                ground_truth
            )
            result['ablation']['full_text'] = ablation_prompt + continuation
            result['ablation']['prompt_diff'] = len(baseline_prompt) - len(ablation_prompt)
        except Exception as e:
            result['ablation']['error'] = str(e)[:200]
        
        try:
            replacement_prompt = self.create_replacement_prompt(
                problem['problem_text'],
                continued_cot_prefix,
                problem['injected_value'],
                problem['original_value']
            )
            
            inputs = self.tokenizer(replacement_prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated_text[len(replacement_prompt):]
            
            result['replacement']['continuation'] = continuation
            result['replacement']['propagated_error'] = self.check_error_propagation(
                replacement_prompt + continuation,
                ground_truth
            )
            result['replacement']['full_text'] = replacement_prompt + continuation
        except Exception as e:
            result['replacement']['error'] = str(e)[:200]
        
        return result
    
    def analyze_dataset(self, dataset_path, max_problems=None):
        """Run causal interventions across the dataset"""
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Causal Intervention Analysis{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        print(f"Loading dataset from {dataset_path}...")
        problems = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        if max_problems:
            problems = problems[:max_problems]
        
        print(f"Loaded {len(problems)} problems\n")
        
        classification_field = 'final_classification' if 'final_classification' in problems[0] else 'classification'
        faithful_problems = [p for p in problems if p[classification_field] is True]
        corrected_problems = [p for p in problems if p[classification_field] is False]
        
        print(f"{C.BOLD}Dataset Composition:{C.ENDC}")
        print(f"  {C.FAIL}Faithful (Propagated Error):{C.ENDC} {len(faithful_problems)} problems")
        print(f"  {C.OKGREEN}Self-Corrected:{C.ENDC} {len(corrected_problems)} problems")
        print(f"  {C.BOLD}Total interventions:{C.ENDC} {(len(faithful_problems) + len(corrected_problems)) * 3}")
        print()
        
        results = {
            'faithful': [],
            'corrected': [],
            'metadata': {
                'model': MODEL_NAME,
                'max_new_tokens': MAX_NEW_TOKENS,
                'total_analyzed': len(faithful_problems) + len(corrected_problems),
                'faithful_count': len(faithful_problems),
                'corrected_count': len(corrected_problems)
            }
        }
        
        print(f"{C.OKCYAN}Running interventions on FAITHFUL problems...{C.ENDC}")
        for i, problem in enumerate(faithful_problems):
            print(f"[{i+1}/{len(faithful_problems)}] Problem {problem['problem_id']} (3 conditions)...", end="\r")
            result = self.run_intervention(problem)
            results['faithful'].append(result)
            
            if (i + 1) % 20 == 0:
                with open(OUTPUT_FILE + '.tmp', 'w') as f:
                    json.dump(results, f, indent=2)
        
        print(" " * 80)
        
        print(f"{C.OKCYAN}Running interventions on SELF-CORRECTED problems...{C.ENDC}")
        for i, problem in enumerate(corrected_problems):
            print(f"[{i+1}/{len(corrected_problems)}] Problem {problem['problem_id']} (3 conditions)...", end="\r")
            result = self.run_intervention(problem)
            results['corrected'].append(result)
            
            if (i + 1) % 20 == 0:
                with open(OUTPUT_FILE + '.tmp', 'w') as f:
                    json.dump(results, f, indent=2)
        
        print(" " * 80)
        
        return results


def compute_intervention_statistics(results):
    """Compute causal effect statistics"""
    
    def count_propagations(group_results, condition):
        """Count how many times error was propagated in a condition"""
        propagated = 0
        corrected = 0
        unclear = 0
        errors = 0
        
        for r in group_results:
            if 'error' in r and condition not in r:
                errors += 1
            elif condition not in r or 'propagated_error' not in r[condition]:
                if 'error' in r[condition]:
                    errors += 1
                else:
                    unclear += 1
            elif r[condition]['propagated_error'] is True:
                propagated += 1
            elif r[condition]['propagated_error'] is False:
                corrected += 1
            else:
                unclear += 1
        
        total = len(group_results)
        valid = propagated + corrected
        return {
            'propagated': propagated,
            'corrected': corrected,
            'unclear': unclear,
            'errors': errors,
            'total': total,
            'valid': valid,
            'propagation_rate': propagated / valid if valid > 0 else None
        }
    
    stats = {
        'faithful': {
            'baseline': count_propagations(results['faithful'], 'baseline'),
            'ablation': count_propagations(results['faithful'], 'ablation'),
            'replacement': count_propagations(results['faithful'], 'replacement')
        },
        'corrected': {
            'baseline': count_propagations(results['corrected'], 'baseline'),
            'ablation': count_propagations(results['corrected'], 'ablation'),
            'replacement': count_propagations(results['corrected'], 'replacement')
        }
    }
    
    return stats


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
        
        analyzer = CausalInterventionAnalyzer(MODEL_NAME, DEVICE)
        
        results = analyzer.analyze_dataset(INPUT_FILE, max_problems=MAX_PROBLEMS)
        
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Causal Effect Analysis{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        stats = compute_intervention_statistics(results)
        
        print(f"{C.BOLD}Error Propagation Rates by Condition:{C.ENDC}\n")
        
        for group_name, group_color, group_stats in [
            ('FAITHFUL', C.FAIL, stats['faithful']),
            ('SELF-CORRECTED', C.OKGREEN, stats['corrected'])
        ]:
            print(f"{group_color}{group_name} Models:{C.ENDC}")
            
            for condition in ['baseline', 'ablation', 'replacement']:
                cond_stats = group_stats[condition]
                rate = cond_stats['propagation_rate']
                
                if rate is not None:
                    print(f"  {condition.capitalize():12} → {rate:5.1%} propagated error "
                          f"({cond_stats['propagated']}/{cond_stats['valid']} valid, "
                          f"{cond_stats['unclear']} unclear, {cond_stats['errors']} errors)")
                else:
                    print(f"  {condition.capitalize():12} → No valid results")
            
            print()
        
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Causal Effects (Change from Baseline){C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        for group_name, group_color, group_stats, results_key in [
            ('FAITHFUL', C.FAIL, stats['faithful'], 'faithful'),
            ('SELF-CORRECTED', C.OKGREEN, stats['corrected'], 'corrected')
        ]:
            baseline_rate = group_stats['baseline']['propagation_rate']
            ablation_rate = group_stats['ablation']['propagation_rate']
            replacement_rate = group_stats['replacement']['propagation_rate']
            
            if baseline_rate is not None and ablation_rate is not None and replacement_rate is not None:
                ablation_effect = ablation_rate - baseline_rate
                replacement_effect = replacement_rate - baseline_rate
                
                print(f"{group_color}{group_name}:{C.ENDC}")
                print(f"  Ablation Effect:    {ablation_effect:+6.1%} (removing error token)")
                print(f"  Replacement Effect: {replacement_effect:+6.1%} (replacing with truth)")
                
                group_baseline = [r['baseline']['propagated_error'] for r in results[results_key] 
                                   if 'baseline' in r and r['baseline'].get('propagated_error') is not None]
                group_ablation = [r['ablation']['propagated_error'] for r in results[results_key] 
                                   if 'ablation' in r and r['ablation'].get('propagated_error') is not None]
                
                if len(group_baseline) > 5 and len(group_ablation) > 5:
                    from scipy.stats import chi2_contingency
                    baseline_prop = sum(group_baseline)
                    ablation_prop = sum(group_ablation)
                    baseline_not = len(group_baseline) - baseline_prop
                    ablation_not = len(group_ablation) - ablation_prop
                    
                    contingency = [[baseline_prop, baseline_not], [ablation_prop, ablation_not]]
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        print(f"  Statistical Test:   χ² = {chi2:.2f}, p = {p_value:.4f}")
                    except:
                        pass
                
                print()
        
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Interpretation{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        faithful_baseline = stats['faithful']['baseline']['propagation_rate']
        faithful_ablation = stats['faithful']['ablation']['propagation_rate']
        corrected_baseline = stats['corrected']['baseline']['propagation_rate']
        corrected_ablation = stats['corrected']['ablation']['propagation_rate']
        
        if faithful_baseline is not None and faithful_ablation is not None:
            faithful_ablation_effect = faithful_ablation - faithful_baseline
            
            print(f"{C.BOLD}Key Findings:{C.ENDC}\n")
            
            if faithful_ablation_effect < -0.2:
                print(f"{C.OKGREEN}✓ FAITHFUL models are CAUSALLY DEPENDENT on error token:{C.ENDC}")
                print(f"  → Removing error reduces propagation by {abs(faithful_ablation_effect):.1%}")
                print(f"  → This SUPPORTS the embedding analysis (they DO read the error)")
                print(f"  → CONTRADICTS attention analysis (low attention but high causal effect)")
                print(f"  → {C.BOLD}Resolution:{C.ENDC} Attention weights don't capture full causal influence")
                print(f"  → The error token has mechanistic importance despite low attention scores")
            elif faithful_ablation_effect > -0.05:
                print(f"{C.WARNING}⚠ FAITHFUL models are NOT causally dependent on error token:{C.ENDC}")
                print(f"  → Removing error barely affects propagation ({faithful_ablation_effect:+.1%})")
                print(f"  → This SUPPORTS the attention analysis (low attention = not reading)")
                print(f"  → CONTRADICTS embedding analysis (aligned representation but not causal)")
                print(f"  → {C.BOLD}Resolution:{C.ENDC} Embedding similarity doesn't imply causal dependence")
                print(f"  → Error propagation happens for other reasons (e.g., chain-of-thought momentum)")
            else:
                print(f"{C.OKCYAN}○ FAITHFUL models show MODERATE dependence on error token:{C.ENDC}")
                print(f"  → Ablation effect: {faithful_ablation_effect:+.1%}")
                print(f"  → Partial evidence for both attention and embedding findings")
                print(f"  → Error token contributes to propagation but is not the sole cause")
            
            print()
        
        if corrected_baseline is not None and corrected_ablation is not None:
            corrected_ablation_effect = corrected_ablation - corrected_baseline
            
            if abs(corrected_ablation_effect) < 0.1:
                print(f"{C.OKGREEN}✓ SELF-CORRECTED models IGNORE the error token:{C.ENDC}")
                print(f"  → Ablation has minimal effect ({corrected_ablation_effect:+.1%})")
                print(f"  → Confirms they were not using the error for reasoning")
                print(f"  → They compute independently of the injected mistake")
            else:
                print(f"{C.WARNING}⚠ SELF-CORRECTED models show UNEXPECTED dependence:{C.ENDC}")
                print(f"  → Ablation effect: {corrected_ablation_effect:+.1%}")
                print(f"  → Suggests they may read the error to actively correct it")
                print(f"  → Correction is an active process, not passive ignorance")
        
        print(f"\n{C.BOLD}Saved detailed results to: {OUTPUT_FILE}{C.ENDC}")
        print(f"{C.BOLD}Temporary checkpoints saved to: {OUTPUT_FILE}.tmp{C.ENDC}")
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