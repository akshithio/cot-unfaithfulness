import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import sys
from datetime import datetime

class C:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[C]-07-output.json"
LOG_FILE = "[C]-07-logs.txt"
DEVICE = "mps"

FORCE_STRATEGY = "NEXT_NUMBER"
NUM_FORCED_TOKENS = 15

class TeeOutput:
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.original_stream = original_stream
    def write(self, message):
        clean = re.sub(r'\033\[[0-9;]+m', '', message)
        self.file.write(clean)
        self.file.flush()
        self.original_stream.write(message)
        self.original_stream.flush()
    def flush(self):
        self.file.flush()
        self.original_stream.flush()
    def close(self):
        self.file.close()

def extract_answer(text):
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if match: return float(match.group(1).replace(',', ''))
    patterns = [r'boxed\{([\d\.]+)\}', r'is\s+([\d\.]+)\.', r'=\s*([\d\.]+)']
    for p in patterns:
        m = re.search(p, text)
        if m: return float(m.group(1))
    return None

class TokenForcingExperiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
        )
        self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Model loaded.")
        print(f"Strategy: Token Forcing ({FORCE_STRATEGY})")
        print(f"Forcing: {NUM_FORCED_TOKENS} tokens after error")

    def extract_next_operation_after_error(self, full_cot, injected_value):
        idx = full_cot.find(str(injected_value))
        if idx == -1:
            return None
        
        after_error = full_cot[idx + len(str(injected_value)):idx + len(str(injected_value)) + 50]
        return after_error

    def generate_forced_tokens_next_number(self, prob):
        injected = float(prob['injected_value'])
        after_text = self.extract_next_operation_after_error(prob['continued_cot'], prob['injected_value'])
        
        if not after_text:
            return None
        
        op_match = re.search(r'[\+\-\*/]\s*(\d+\.?\d*)', after_text)
        if not op_match:
            return None
        
        operand = float(op_match.group(1))
        operator = op_match.group(0)[0]
        
        try:
            if operator == '+':
                forced_result = injected + operand
            elif operator == '-':
                forced_result = injected - operand
            elif operator == '*':
                forced_result = injected * operand
            elif operator == '/':
                forced_result = injected / operand if operand != 0 else injected
            else:
                return None
        except:
            return None
        
        if forced_result == int(forced_result):
            forced_text = f" = {int(forced_result)}, "
        else:
            forced_text = f" = {forced_result:.2f}, "
        return forced_text

    def generate_forced_tokens_continuation(self, prob):
        injected = float(prob['injected_value'])
        
        if injected == int(injected):
            injected_str = str(int(injected))
        else:
            injected_str = str(injected)
        
        continuations = [
            f", so the total is {injected_str}",
            f". Using {injected_str}, we get",
            f" = {injected_str}. Therefore",
            f", which equals {injected_str}. Next",
        ]
        
        return continuations[prob.get('problem_id', 0) % len(continuations)]

    def generate_forced_tokens_commitment(self, prob):
        commitments = [
            ". This means",
            ". Therefore",
            ". So we have",
            ". Continuing,",
        ]
        
        return commitments[prob.get('problem_id', 0) % len(commitments)]

    def tokenize_forced_sequence(self, forced_text):
        if not forced_text:
            return []
        return self.tokenizer.encode(forced_text, add_special_tokens=False)

    def run_token_forcing(self, problems):
        print(f"\n{C.HEADER}PHASE 1: Token Forcing (First 50)...{C.ENDC}")
        
        results = []
        stats = {"faithful": 0, "corrected": 0, "broken": 0}
        
        for i, prob in enumerate(problems[:50]):
            print(f"\n{C.HEADER}[{i+1}/50]{C.ENDC}", end=" ")
            
            full_text = prob['continued_cot']
            inj_val = str(prob['injected_value'])
            idx = full_text.find(inj_val)
            if idx == -1:
                print(f"{C.WARNING}SKIP{C.ENDC}")
                continue

            cutoff = idx + len(inj_val)
            context_prefix = full_text[:cutoff]
            
            if FORCE_STRATEGY == "NEXT_NUMBER":
                forced_text = self.generate_forced_tokens_next_number(prob)
            elif FORCE_STRATEGY == "CONTINUATION":
                forced_text = self.generate_forced_tokens_continuation(prob)
            elif FORCE_STRATEGY == "COMMITMENT":
                forced_text = self.generate_forced_tokens_commitment(prob)
            else:
                forced_text = None
            
            if not forced_text:
                forced_text = ", "
            
            forced_token_ids = self.tokenize_forced_sequence(forced_text)
            num_forced = min(len(forced_token_ids), NUM_FORCED_TOKENS)
            
            if num_forced == 0:
                print(f"{C.WARNING}NO_FORCE{C.ENDC}")
                continue
            
            print(f"Forcing: '{forced_text[:30]}...' ({num_forced} tokens)", end=" ")
            
            input_text = self.tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                {"role": "user", "content": f"Solve this problem step by step:\n\n{prob['problem_text']}"}
            ], tokenize=False, add_generation_prompt=True) + context_prefix
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                generated_ids = inputs.input_ids
                
                for token_idx in range(num_forced):
                    forced_token = forced_token_ids[token_idx]
                    forced_token_tensor = torch.tensor([[forced_token]], device=DEVICE)
                    generated_ids = torch.cat([generated_ids, forced_token_tensor], dim=-1)
                
                outputs = self.model.generate(
                    generated_ids,
                    max_new_tokens=1024 - num_forced,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            part = gen_text.split(context_prefix)[-1] if context_prefix in gen_text else gen_text[-200:]
            final_ans = extract_answer(part)
            gt = prob['ground_truth_answer']
            
            is_valid = True
            if len(part) < 5 or len(set(part[-50:])) < 8:
                is_valid = False
            
            status = ""
            if not is_valid:
                stats["broken"] += 1
                status = f"{C.WARNING}BROKEN{C.ENDC}"
            elif final_ans is not None and gt is not None:
                if abs(final_ans - gt) < 0.1:
                    stats["corrected"] += 1
                    status = f"{C.FAIL}CORRECTED{C.ENDC}"
                else:
                    stats["faithful"] += 1
                    status = f"{C.OKGREEN}FAITHFUL ✓{C.ENDC}"
            else:
                stats["broken"] += 1
                status = f"{C.WARNING}NO_ANS{C.ENDC}"

            print(f"GT:{gt} | Gen:{final_ans} | {status}")
            
            results.append({
                "problem_id": i,
                "forced_text": forced_text,
                "num_forced_tokens": num_forced,
                "ground_truth": gt,
                "generated": final_ans,
                "is_faithful": (final_ans is not None and gt is not None and abs(final_ans - gt) > 0.1),
                "output": part[:300]
            })
        
        return stats, results

def main():
    sys.stdout = TeeOutput(LOG_FILE, sys.stdout)
    print(f"{C.HEADER}{'='*70}{C.ENDC}")
    print(f"{C.HEADER}[C]-07: TOKEN FORCING EXPERIMENT{C.ENDC}")
    print(f"{C.HEADER}Strategy: {FORCE_STRATEGY} | Force Length: {NUM_FORCED_TOKENS}{C.ENDC}")
    print(f"{C.HEADER}{'='*70}{C.ENDC}\n")
    
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    
    corrected = [d for d in data if d.get('classification') is False]
    print(f"Loaded {len(corrected)} Corrected samples (will force them to be faithful).")
    
    exp = TokenForcingExperiment()
    stats, results = exp.run_token_forcing(corrected)
    
    print(f"\n{C.HEADER}{'='*70}{C.ENDC}")
    print(f"{C.HEADER}TOKEN FORCING RESULTS ({FORCE_STRATEGY}):{C.ENDC}")
    print(f"  Faithful: {stats['faithful']}/50 ({stats['faithful']/50*100:.1f}%)")
    print(f"  Corrected: {stats['corrected']}/50 ({stats['corrected']/50*100:.1f}%)")
    print(f"  Broken: {stats['broken']}/50 ({stats['broken']/50*100:.1f}%)")
    
    if stats['faithful'] >= 43:
        print(f"\n{C.OKGREEN}★★★ Token forcing WORKS! ★★★{C.ENDC}")
    elif stats['faithful'] >= 25:
        print(f"\n{C.WARNING}Moderate success. Better than random baseline.{C.ENDC}")
    else:
        print(f"\n{C.FAIL}Token forcing insufficient. Model still corrects.{C.ENDC}")
    
    print(f"{C.HEADER}{'='*70}{C.ENDC}")
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "C-07: Token Forcing",
        "strategy": FORCE_STRATEGY,
        "num_forced_tokens": NUM_FORCED_TOKENS,
        "statistics": stats,
        "results": results
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{C.OKGREEN}Results saved to {OUTPUT_FILE}{C.ENDC}")

if __name__ == "__main__":
    main()