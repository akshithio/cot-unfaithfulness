"""
[B]-06 component_ablation.py:

Phase B: Mechanistic EDA
Part 06: Component-Wise Causal Ablation (Attention vs MLP)

This script resolves the attention paradox by decomposing the causal pathway:
- Low attention weights but high causal effect
- Is the error propagating via attention mechanism or residual stream?

Methodology:
For each problem, run FIVE conditions:
1. BASELINE: Normal generation with error present
2. ABLATE_ATTENTION: Zero out attention outputs at error position (keep MLP)
3. ABLATE_MLP: Zero out MLP outputs at error position (keep attention)
4. ABLATE_BOTH: Zero out both (equivalent to previous full ablation)
5. REPLACEMENT: Replace error with correct value (control)

Key Questions:
- If attention ablation rescues 90% → attention IS the pathway (paradox!)
- If MLP ablation rescues 90% → residual stream IS the pathway ✓
- If both needed → multiple pathways
- Test additivity: ablate_both ≈ ablate_attention + ablate_MLP?

Input: [A].jsonl (dataset with injected errors)
Output: [B]-06-output.json, [B]-05-logs.txt
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

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[B]-06-output.json"
LOG_FILE = "[B]-06-logs.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MAX_PROBLEMS = None
MAX_NEW_TOKENS = 1024
LAYERS_TO_INTERVENE = 28

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


class ComponentAblationHook:
    """Hook to ablate specific components (attention/MLP) at error positions"""
    
    def __init__(self, error_positions, ablate_attention=False, ablate_mlp=False):
        self.error_positions = error_positions
        self.ablate_attention = ablate_attention
        self.ablate_mlp = ablate_mlp
        self.hooks = []
    
    def create_attention_hook(self, layer_idx):
        """Hook to zero out attention output at error positions"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output
            
            modified_output = attn_output.clone()
            
            for pos in self.error_positions:
                if pos < modified_output.shape[1]:
                    modified_output[:, pos, :] = 0.0
            
            if isinstance(output, tuple):
                return (modified_output,) + output[1:]
            return modified_output
        
        return hook
    
    def create_mlp_hook(self, layer_idx):
        """Hook to zero out MLP output at error positions"""
        def hook(module, input, output):
            modified_output = output.clone()
            
            for pos in self.error_positions:
                if pos < modified_output.shape[1]:
                    modified_output[:, pos, :] = 0.0
            
            return modified_output
        
        return hook
    
    def register_hooks(self, model, num_layers_to_intervene):
        """Register hooks on the model"""
        num_layers = len(model.model.layers)
        start_layer = max(0, num_layers - num_layers_to_intervene)
        
        for layer_idx in range(start_layer, num_layers):
            layer = model.model.layers[layer_idx]
            
            if self.ablate_attention and hasattr(layer.self_attn, 'o_proj'):
                hook = layer.self_attn.o_proj.register_forward_hook(
                    self.create_attention_hook(layer_idx)
                )
                self.hooks.append(hook)
            
            if self.ablate_mlp and hasattr(layer, 'mlp'):
                if hasattr(layer.mlp, 'down_proj'):
                    hook = layer.mlp.down_proj.register_forward_hook(
                        self.create_mlp_hook(layer_idx)
                    )
                    self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class ComponentAblationAnalyzer:
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
        print(f"  Layers to intervene: {LAYERS_TO_INTERVENE}")
        print(f"  Max new tokens: {MAX_NEW_TOKENS}")
        print()
    
    def extract_final_answer(self, text):
        """Extract numerical answer from text"""
        answer_patterns = [
            (r'\\boxed\{\s*\\frac\{(\d+)\}\{(\d+)\}\s*\}', 'frac'),
            (r'\\boxed\{\s*(\d+)\\frac\{(\d+)\}\{(\d+)\}\s*\}', 'mixed'),
            (r'\\boxed\{\s*(\$?-?[\d,]+(?:\.\d+)?).*?\}', 'decimal'),
            (r'\\frac\{(\d+)\}\{(\d+)\}', 'frac'),
            (r'(?:answer|result)(?:\s+is)?[:\s]+(\d+)/(\d+)', 'text_frac'),
            (r'(?:final answer|answer|result)(?:\s+is)?[:\s]+\$?\s*([\d,]+(?:\.\d+)?)', 'decimal'),
            (r'####\s*([\d,]+(?:\.\d+)?)', 'decimal'),
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
        """Check if error propagated (True) or was corrected (False)"""
        final_answer = self.extract_final_answer(generated_text)
        
        if final_answer is None:
            return None
        
        ground_truth = float(ground_truth)
        is_correct = abs(final_answer - ground_truth) < 0.01 * abs(ground_truth) + 0.01
        
        return not is_correct
    
    def find_error_positions(self, full_text, injected_value):
        """Find token positions of injected error"""
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        injected_tokens = self.tokenizer.encode(str(injected_value), add_special_tokens=False)
        
        positions = []
        for i in range(len(tokens) - len(injected_tokens) + 1):
            if tokens[i:i+len(injected_tokens)] == injected_tokens:
                positions.extend(range(i, i + len(injected_tokens)))
        
        return positions
    
    def generate_with_ablation(self, prompt, error_positions, 
                               ablate_attention=False, ablate_mlp=False):
        """Generate text with selective component ablation"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(self.device)
        
        hook_manager = ComponentAblationHook(
            error_positions=error_positions,
            ablate_attention=ablate_attention,
            ablate_mlp=ablate_mlp
        )
        
        try:
            hook_manager.register_hooks(self.model, LAYERS_TO_INTERVENE)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated_text[len(prompt):]
            
            return continuation, generated_text
            
        finally:
            hook_manager.remove_hooks()
    
    def run_intervention(self, problem):
        """Run all five intervention conditions"""
        result = {
            'problem_id': problem['problem_id'],
            'injected_value': problem['injected_value'],
            'original_value': problem['original_value'],
            'ground_truth': problem.get('ground_truth_answer'),
            'baseline': {},
            'ablate_attention': {},
            'ablate_mlp': {},
            'ablate_both': {},
            'replacement': {}
        }
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
            {"role": "user", "content": f"Solve this problem step by step:\n\n{problem['problem_text']}"}
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        continued_cot = problem['continued_cot']
        injection_pos = continued_cot.find(str(problem['injected_value']))
        
        if injection_pos == -1:
            result['error'] = 'Could not find injection position'
            return result
        
        prefix_end = min(injection_pos + len(str(problem['injected_value'])) + 100, len(continued_cot))
        continued_cot_prefix = continued_cot[:prefix_end]
        
        full_prompt = base_prompt + continued_cot_prefix
        
        error_positions = self.find_error_positions(continued_cot_prefix, problem['injected_value'])
        
        if not error_positions:
            result['error'] = 'Could not find error tokens'
            return result
        
        base_length = len(self.tokenizer.encode(base_prompt, add_special_tokens=False))
        error_positions = [pos + base_length for pos in error_positions]
        
        result['error_positions'] = error_positions
        result['prefix_length'] = len(continued_cot_prefix)
        
        try:
            continuation, full_text = self.generate_with_ablation(
                full_prompt, error_positions, 
                ablate_attention=False, ablate_mlp=False
            )
            result['baseline']['continuation'] = continuation
            result['baseline']['propagated_error'] = self.check_error_propagation(
                full_text, problem.get('ground_truth_answer')
            )
        except Exception as e:
            result['baseline']['error'] = str(e)[:200]
        
        try:
            continuation, full_text = self.generate_with_ablation(
                full_prompt, error_positions,
                ablate_attention=True, ablate_mlp=False
            )
            result['ablate_attention']['continuation'] = continuation
            result['ablate_attention']['propagated_error'] = self.check_error_propagation(
                full_text, problem.get('ground_truth_answer')
            )
        except Exception as e:
            result['ablate_attention']['error'] = str(e)[:200]
        
        try:
            continuation, full_text = self.generate_with_ablation(
                full_prompt, error_positions,
                ablate_attention=False, ablate_mlp=True
            )
            result['ablate_mlp']['continuation'] = continuation
            result['ablate_mlp']['propagated_error'] = self.check_error_propagation(
                full_text, problem.get('ground_truth_answer')
            )
        except Exception as e:
            result['ablate_mlp']['error'] = str(e)[:200]
        
        try:
            continuation, full_text = self.generate_with_ablation(
                full_prompt, error_positions,
                ablate_attention=True, ablate_mlp=True
            )
            result['ablate_both']['continuation'] = continuation
            result['ablate_both']['propagated_error'] = self.check_error_propagation(
                full_text, problem.get('ground_truth_answer')
            )
        except Exception as e:
            result['ablate_both']['error'] = str(e)[:200]
        
        try:
            replacement_prompt = full_prompt.replace(
                str(problem['injected_value']), 
                str(problem['original_value'])
            )
            continuation, full_text = self.generate_with_ablation(
                replacement_prompt, [],
                ablate_attention=False, ablate_mlp=False
            )
            result['replacement']['continuation'] = continuation
            result['replacement']['propagated_error'] = self.check_error_propagation(
                full_text, problem.get('ground_truth_answer')
            )
        except Exception as e:
            result['replacement']['error'] = str(e)[:200]
        
        return result
    
    def analyze_dataset(self, dataset_path, max_problems=None):
        """Run component ablations across dataset"""
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Component-Wise Causal Ablation Analysis{C.ENDC}")
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
        print(f"  Faithful: {len(faithful_problems)} problems")
        print(f"  Self-Corrected: {len(corrected_problems)} problems")
        print(f"  Total interventions: {(len(faithful_problems) + len(corrected_problems)) * 5}")
        print()
        
        results = {
            'faithful': [],
            'corrected': [],
            'metadata': {
                'model': MODEL_NAME,
                'layers_intervened': LAYERS_TO_INTERVENE,
                'max_new_tokens': MAX_NEW_TOKENS
            }
        }
        
        print(f"{C.OKCYAN}Running interventions on FAITHFUL problems...{C.ENDC}")
        for i, problem in enumerate(faithful_problems):
            print(f"[{i+1}/{len(faithful_problems)}] Problem {problem['problem_id']} (5 conditions)...", end="\r")
            result = self.run_intervention(problem)
            results['faithful'].append(result)
            
            if (i + 1) % 10 == 0:
                with open(OUTPUT_FILE + '.tmp', 'w') as f:
                    json.dump(results, f, indent=2)
        
        print(" " * 80)
        
        print(f"{C.OKCYAN}Running interventions on SELF-CORRECTED problems...{C.ENDC}")
        for i, problem in enumerate(corrected_problems):
            print(f"[{i+1}/{len(corrected_problems)}] Problem {problem['problem_id']} (5 conditions)...", end="\r")
            result = self.run_intervention(problem)
            results['corrected'].append(result)
            
            if (i + 1) % 10 == 0:
                with open(OUTPUT_FILE + '.tmp', 'w') as f:
                    json.dump(results, f, indent=2)
        
        print(" " * 80)
        
        return results


def compute_statistics(results):
    """Compute comprehensive statistics"""
    
    def count_propagations(group_results, condition):
        propagated, corrected, unclear, errors = 0, 0, 0, 0
        
        for r in group_results:
            if condition not in r or 'propagated_error' not in r[condition]:
                if condition in r and 'error' in r[condition]:
                    errors += 1
                else:
                    unclear += 1
            elif r[condition]['propagated_error'] is True:
                propagated += 1
            elif r[condition]['propagated_error'] is False:
                corrected += 1
            else:
                unclear += 1
        
        valid = propagated + corrected
        return {
            'propagated': propagated,
            'corrected': corrected,
            'unclear': unclear,
            'errors': errors,
            'valid': valid,
            'rate': propagated / valid if valid > 0 else None
        }
    
    stats = {}
    for group in ['faithful', 'corrected']:
        stats[group] = {
            'baseline': count_propagations(results[group], 'baseline'),
            'ablate_attention': count_propagations(results[group], 'ablate_attention'),
            'ablate_mlp': count_propagations(results[group], 'ablate_mlp'),
            'ablate_both': count_propagations(results[group], 'ablate_both'),
            'replacement': count_propagations(results[group], 'replacement')
        }
    
    return stats


def main():
    original_stdout = sys.stdout
    tee_stdout = TeeOutput(LOG_FILE, original_stdout)
    sys.stdout = tee_stdout
    
    try:
        print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {LOG_FILE}\n")
        
        analyzer = ComponentAblationAnalyzer(MODEL_NAME, DEVICE)
        results = analyzer.analyze_dataset(INPUT_FILE, max_problems=MAX_PROBLEMS)
        
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        stats = compute_statistics(results)
        
        print(f"\n{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Component Ablation Results{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        for group_name, group_data in [('FAITHFUL', stats['faithful']), 
                                       ('SELF-CORRECTED', stats['corrected'])]:
            print(f"{C.BOLD}{group_name}:{C.ENDC}")
            
            for cond in ['baseline', 'ablate_attention', 'ablate_mlp', 'ablate_both', 'replacement']:
                s = group_data[cond]
                if s['rate'] is not None:
                    print(f"  {cond:18} → {s['rate']:5.1%} propagated "
                          f"({s['propagated']}/{s['valid']} valid)")
            
            baseline = group_data['baseline']['rate']
            if baseline is not None:
                print(f"\n  {C.BOLD}Rescue Effects (vs baseline):{C.ENDC}")
                for cond in ['ablate_attention', 'ablate_mlp', 'ablate_both']:
                    rate = group_data[cond]['rate']
                    if rate is not None:
                        effect = rate - baseline
                        print(f"    {cond:18}: {effect:+6.1%}")
                
                attn_rate = group_data['ablate_attention']['rate']
                mlp_rate = group_data['ablate_mlp']['rate']
                both_rate = group_data['ablate_both']['rate']
                
                if all(r is not None for r in [attn_rate, mlp_rate, both_rate]):
                    attn_effect = attn_rate - baseline
                    mlp_effect = mlp_rate - baseline
                    predicted_combined = attn_effect + mlp_effect
                    actual_combined = both_rate - baseline
                    
                    print(f"\n  {C.BOLD}Additivity Test:{C.ENDC}")
                    print(f"    Attention effect:  {attn_effect:.1%}")
                    print(f"    MLP effect:        {mlp_effect:.1%}")
                    print(f"    Sum (predicted):   {predicted_combined:.1%}")
                    print(f"    Both (actual):     {actual_combined:.1%}")
                    print(f"    Difference:        {abs(predicted_combined - actual_combined):.1%}")
                    
                    if abs(predicted_combined - actual_combined) < 0.05:
                        print(f"    → {C.OKGREEN}Approximately additive{C.ENDC}")
                    else:
                        print(f"    → {C.WARNING}Non-additive (interaction effects){C.ENDC}")
            
            print()
        
        print(f"{C.HEADER}{'='*80}{C.ENDC}")
        print(f"{C.HEADER}Interpretation{C.ENDC}")
        print(f"{C.HEADER}{'='*80}{C.ENDC}\n")
        
        faithful = stats['faithful']
        if faithful['baseline']['rate'] and faithful['ablate_mlp']['rate'] and faithful['ablate_attention']['rate']:
            baseline_rate = faithful['baseline']['rate']
            mlp_rescue = baseline_rate - faithful['ablate_mlp']['rate']
            attn_rescue = baseline_rate - faithful['ablate_attention']['rate'] 
            
            print(f"{C.BOLD}Key Findings:{C.ENDC}\n")
            
            if mlp_rescue > 0.7 and attn_rescue < 0.3:
                print(f"{C.OKGREEN}✓ ERROR PROPAGATES VIA RESIDUAL STREAM (MLP):{C.ENDC}")
                print(f"  → MLP ablation rescues {mlp_rescue:.1%} of errors")
                print(f"  → Attention ablation only rescues {attn_rescue:.1%}")
                print(f"  → This RESOLVES the paradox:")
                print(f"    • Low attention weights are accurate")
                print(f"    • Error uses residual stream, not attention mechanism")
                print(f"    • MLP writes directly to residual, bypassing attention")
            elif attn_rescue > 0.7 and mlp_rescue < 0.3:
                print(f"{C.WARNING}✓ ERROR PROPAGATES VIA ATTENTION:{C.ENDC}")
                print(f"  → Attention ablation rescues {attn_rescue:.1%}")
                print(f"  → This creates a deeper paradox:")
                print(f"    • Low attention weights but high causal effect")
                print(f"    • Suggests: small weights × large values = big impact")
                print(f"    • Or: specific attention heads matter more than average")
            elif mlp_rescue > 0.4 and attn_rescue > 0.4:
                print(f"{C.OKCYAN}○ DUAL PATHWAY:{C.ENDC}")
                print(f"  → Both attention ({attn_rescue:.1%}) and MLP ({mlp_rescue:.1%}) contribute")
                print(f"  → Error uses multiple mechanisms")
            
        print(f"\n{C.OKGREEN}Analysis complete!{C.ENDC}")
        print(f"Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        sys.stdout = original_stdout
        tee_stdout.close()


if __name__ == "__main__":
    main()