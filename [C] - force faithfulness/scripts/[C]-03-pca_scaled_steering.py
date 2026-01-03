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
    BOLD = '\033[1m'

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[C]-03-output.json"
LOG_FILE = "[C]-03-logs.txt"
DEVICE = "mps"

NUM_HARVEST_SAMPLES = 50
NUM_TEST_SAMPLES = 50
INJECTION_LAYERS = range(15, 25)
STRENGTH_MULTIPLIER = 0.6

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

def is_coherent(text):
    if not text or len(text) < 10: return False
    if len(set(text[-50:])) < 12: return False
    if text.count('\\') > 30: return False
    return True

class PCAExperiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
        )
        self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded. Targeting Layers {min(INJECTION_LAYERS)}-{max(INJECTION_LAYERS)}.")

    def get_hidden_states_at_error(self, problems):
        layer_states = {l: [] for l in range(self.num_layers)}
        print(f"  Harvesting {len(problems)} samples...")
        
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

    def compute_pca_vectors(self, faithful, corrected):
        print(f"\n{C.HEADER}PHASE 1: Harvesting & Computing PCA Vectors...{C.ENDC}")
        
        f_states = self.get_hidden_states_at_error(faithful)
        c_states = self.get_hidden_states_at_error(corrected)
        
        vectors = {}
        magnitudes = {}
        
        print(f"{C.BOLD}Layer | Natural Norm | Explained Var (PC1){C.ENDC}")
        
        for l in range(self.num_layers):
            if not f_states[l] or not c_states[l]: continue
            
            f_stack = torch.stack(f_states[l]).float()
            c_stack = torch.stack(c_states[l]).float()
            
            f_mean = f_stack.mean(dim=0)
            c_mean = c_stack.mean(dim=0)
            raw_vec = f_mean - c_mean
            
            natural_norm = torch.norm(raw_vec).item()
            
            direction = raw_vec / (natural_norm + 1e-8)
            
            final_vec = direction * (natural_norm * STRENGTH_MULTIPLIER)
            
            vectors[l] = final_vec.to(dtype=self.model.dtype, device=DEVICE)
            
            if l in INJECTION_LAYERS:
                print(f"L{l:02d}  | {natural_norm:8.4f}     | Injection Strength: {natural_norm * STRENGTH_MULTIPLIER:.4f}")
            
        print(f"  Computed {len(vectors)} scaled vectors.")
        return vectors

    def run_intervention(self, problems, vectors):
        print(f"\n{C.HEADER}PHASE 2: Scaled Intervention...{C.ENDC}")
        print(f"  Injecting into Layers {min(INJECTION_LAYERS)}-{max(INJECTION_LAYERS)}")
        print(f"  Multiplier: {STRENGTH_MULTIPLIER}x Natural Norm")

        def get_hook(layer_idx, vec):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden = hidden + vec
                    return (hidden,) + output[1:]
                else:
                    return output + vec
            return hook

        hooks = []
        for l in INJECTION_LAYERS:
            if l in vectors:
                h = self.model.model.layers[l].register_forward_hook(get_hook(l, vectors[l]))
                hooks.append(h)

        print(f"  Hooks registered. Generating...\n")
        
        results = []
        success_cnt = 0
        coherent_cnt = 0
        
        try:
            for i, prob in enumerate(problems):
                full_text = prob['continued_cot']
                inj_val = str(prob['injected_value'])
                idx = full_text.find(inj_val)
                if idx == -1: continue

                cutoff = idx + len(inj_val)
                context_prefix = full_text[:cutoff]
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                    {"role": "user", "content": f"Solve this problem step by step:\n\n{prob['problem_text']}"}
                ]
                base_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_text = base_prompt + context_prefix
                inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if context_prefix in generated_full:
                    part = generated_full.split(context_prefix)[-1]
                else:
                    part = generated_full[-500:]

                final_ans = extract_answer(part)
                gt = prob['ground_truth_answer']
                
                is_coherent_val = is_coherent(part)
                is_faithful = False
                
                status = f"{C.FAIL}STILL CORRECTED{C.ENDC}"
                
                if not is_coherent_val:
                    status = f"{C.WARNING}INCOHERENT{C.ENDC}"
                elif final_ans is not None and gt is not None:
                    if abs(final_ans - gt) > 0.1: 
                        is_faithful = True
                        status = f"{C.OKGREEN}FORCED FAITHFUL{C.ENDC}"
                        success_cnt += 1
                        coherent_cnt += 1
                    else:
                        coherent_cnt += 1

                print(f"[{i+1}/{len(problems)}] GT:{gt} | Gen:{final_ans} | {status}")
                if not is_coherent_val:
                    print(f"   Sample: {part[:60]}...")

                results.append({
                    "problem_id": prob['problem_id'],
                    "is_faithful": is_faithful,
                    "is_coherent": is_coherent_val,
                    "text": part
                })

        finally:
            for h in hooks: h.remove()
            
        return success_cnt, coherent_cnt, len(results), results

def main():
    sys.stdout = TeeOutput(LOG_FILE, sys.stdout)
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
            
    faithful = [d for d in data if d.get('classification') is True]
    corrected = [d for d in data if d.get('classification') is False]
    
    if len(faithful) < 10: return

    exp = PCAExperiment()
    vectors = exp.compute_pca_vectors(faithful[:NUM_HARVEST_SAMPLES], corrected[:NUM_HARVEST_SAMPLES])
    
    test_set = corrected[NUM_HARVEST_SAMPLES:NUM_HARVEST_SAMPLES+NUM_TEST_SAMPLES]
    if not test_set: test_set = corrected[:NUM_TEST_SAMPLES]
    
    success, coherent_total, total, res = exp.run_intervention(test_set, vectors)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(res, f, indent=2)
        
    print(f"\nRESULTS: {success}/{total} ({success/total:.1%}) Forced Faithful")
    print(f"Coherent: {coherent_total}/{total}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()