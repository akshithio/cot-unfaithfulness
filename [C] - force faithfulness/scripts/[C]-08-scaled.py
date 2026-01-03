import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import sys
from tqdm import tqdm

class C:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[C]-08-output.json"
LOG_FILE = "[C]-08-logs.txt"
DEVICE = "mps"

INJECTION_LAYERS = range(12, 25) 
STEERING_COEFF = 20.0
HARVEST_SIZE = 100 

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

class Experiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
        )
        self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded. Configuration: (L12-24, Coeff {STEERING_COEFF})")

    def get_hidden_states_at_error(self, problems):
        layer_states = {l: [] for l in range(self.num_layers)}
        print(f"  Harvesting vectors from {len(problems)} samples...")
        
        for i, prob in enumerate(problems):
            full_text = prob['continued_cot']
            inj_val = str(prob['injected_value'])
            idx = full_text.find(inj_val)
            if idx == -1: continue
            
            cutoff = idx + len(inj_val)
            prefix = full_text[:cutoff]
            inputs = self.tokenizer(prefix, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            for l in range(self.num_layers):
                state = outputs.hidden_states[l+1][0, -1, :].cpu()
                layer_states[l].append(state)
        return layer_states

    def compute_vectors(self, faithful, corrected):
        print(f"\n{C.HEADER}PHASE 1: Computing Steering Vectors...{C.ENDC}")
        f_states = self.get_hidden_states_at_error(faithful)
        c_states = self.get_hidden_states_at_error(corrected)
        
        vectors = {}
        for l in range(self.num_layers):
            if not f_states[l] or not c_states[l]: continue
            f_stack = torch.stack(f_states[l]).float()
            c_stack = torch.stack(c_states[l]).float()
            
            vec = f_stack.mean(dim=0) - c_stack.mean(dim=0)
            
            norm = torch.norm(vec)
            if norm > 1e-6: vec = vec / norm
            
            vectors[l] = vec.to(dtype=self.model.dtype, device=DEVICE)
            
        print(f"  Computed normalized steering vectors for {len(vectors)} layers.")
        return vectors

    def find_target_indices(self, input_ids_list, error_str):
        n_tokens = len(input_ids_list)
        for k in range(1, 20): 
            suffix_ids = input_ids_list[-k:]
            decoded = self.tokenizer.decode(suffix_ids)
            if error_str in decoded:
                start_idx = n_tokens - k
                buffer_idx = max(0, start_idx - 1)
                return list(range(buffer_idx, n_tokens))
        
        return [n_tokens-1, n_tokens-2]

    def run_full_scale(self, problems, vectors):
        print(f"\n{C.HEADER}PHASE 2: Full Scale Intervention...{C.ENDC}")
        print(f"Targeting {len(problems)} self-correcting examples.")
        
        results = []
        stats = {"faithful": 0, "corrected": 0, "broken": 0}
        
        class State:
            target_indices = None
        state = State()

        def get_hook(layer_idx, vec):
            def hook(module, args, output):
                if isinstance(output, tuple): hidden = output[0]
                else: hidden = output
                
                seq_len = hidden.shape[1]
                
                if seq_len > 1 and state.target_indices:
                    v = vec.to(dtype=hidden.dtype, device=hidden.device)
                    for idx in state.target_indices[0]: 
                        if idx < seq_len:
                            hidden[0, idx, :] += (v * STEERING_COEFF)
                
                if isinstance(output, tuple): return (hidden,) + output[1:]
                return hidden
            return hook

        hooks = []
        for l in INJECTION_LAYERS:
            if l in vectors:
                h = self.model.model.layers[l].register_forward_hook(get_hook(l, vectors[l]))
                hooks.append(h)

        print(f"{C.WARNING}Steering Active: Layers {min(INJECTION_LAYERS)}-{max(INJECTION_LAYERS)} | Strength {STEERING_COEFF}{C.ENDC}")

        try:
            for i, prob in enumerate(tqdm(problems, desc="Steering")):
                full_text = prob['continued_cot']
                inj_val = str(prob['injected_value'])
                
                idx = full_text.find(inj_val)
                if idx == -1: 
                    results.append({"id": prob['problem_id'], "status": "skipped"})
                    continue

                cutoff = idx + len(inj_val)
                context_prefix = full_text[:cutoff]
                
                input_text = self.tokenizer.apply_chat_template([
                    {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                    {"role": "user", "content": f"Solve this problem step by step:\n\n{prob['problem_text']}"}
                ], tokenize=False, add_generation_prompt=True) + context_prefix
                
                inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
                input_ids = inputs.input_ids[0].tolist()
                
                target_idxs = self.find_target_indices(input_ids, inj_val)
                state.target_indices = [target_idxs]
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=1024, do_sample=False, 
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                part = generated_full.split(context_prefix)[-1] if context_prefix in generated_full else generated_full[-200:]
                
                final_ans = extract_answer(part)
                gt = prob['ground_truth_answer']
                
                status = "UNKNOWN"
                if len(part) < 10 or len(set(part[-50:])) < 10:
                    status = "BROKEN"
                    stats['broken'] += 1
                elif final_ans is not None and gt is not None:
                    if abs(final_ans - gt) > 0.1:
                        status = "FAITHFUL"
                        stats['faithful'] += 1
                    else:
                        status = "CORRECTED"
                        stats['corrected'] += 1
                else:
                    status = "BROKEN" 
                    stats['broken'] += 1

                results.append({
                    "problem_id": prob['problem_id'],
                    "status": status,
                    "final_answer": final_ans,
                    "ground_truth": gt,
                    "generated_text": part
                })
                
                if i < 5 or i % 10 == 0:
                    color = C.OKGREEN if status == "FAITHFUL" else C.FAIL if status == "CORRECTED" else C.WARNING
                    print(f" [{i+1}] {color}{status}{C.ENDC} | GT:{gt} | Gen:{final_ans}")

        finally:
            for h in hooks: h.remove()
            
        return stats, results

def main():
    sys.stdout = TeeOutput(LOG_FILE, sys.stdout)
 
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
            
    faithful_set = [d for d in data if d.get('classification') is True]
    corrected_set = [d for d in data if d.get('classification') is False]
    
    print(f"Data Loaded: {len(faithful_set)} Faithful, {len(corrected_set)} Corrected.")
    
    exp = Experiment()
    harvest_f = faithful_set[:HARVEST_SIZE]
    harvest_c = corrected_set[:HARVEST_SIZE]
    vectors = exp.compute_vectors(harvest_f, harvest_c)
    
    stats, results = exp.run_full_scale(corrected_set, vectors)
    
    total = len(results)
    success_rate = (stats['faithful'] / total * 100) if total > 0 else 0
    
    print(f"\n{C.HEADER}{'='*60}{C.ENDC}")
    print(f"{C.HEADER}FINAL RESULTS ([C]-31){C.ENDC}")
    print(f"{C.HEADER}{'='*60}{C.ENDC}")
    print(f"Total Samples: {total}")
    print(f"Faithful (Success): {stats['faithful']} ({success_rate:.1f}%)")
    print(f"Corrected (Fail):   {stats['corrected']} ({stats['corrected']/total*100:.1f}%)")
    print(f"Broken/Incoherent:  {stats['broken']} ({stats['broken']/total*100:.1f}%)")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            "config": {"layers": str(INJECTION_LAYERS), "coeff": STEERING_COEFF, "method": "Scaling"},
            "stats": stats,
            "results": results
        }, f, indent=2)
    
    print(f"Detailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()