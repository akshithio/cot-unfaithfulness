import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import sys
from datetime import datetime

class C:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    OKCYAN = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[C]-02-output.json"
LOG_FILE = "[C]-02-logs.txt"
DEVICE = "mps"

NUM_HARVEST_SAMPLES = 50
NUM_TEST_SAMPLES = 50
STEERING_COEFF = 1.5
INJECTION_LAYERS = range(10, 26)

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


def extract_answer(text):
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', text)
    if match: return float(match.group(1).replace(',', ''))
    patterns = [r'boxed\{([\d\.]+)\}', r'is\s+([\d\.]+)\.', r'=\s*([\d\.]+)']
    for p in patterns:
        m = re.search(p, text)
        if m: return float(m.group(1))
    return None


class SteeringExperiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
        )
        self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded: {self.num_layers} layers.")

    def get_hidden_states_at_error(self, problems):
        layer_states = {l: [] for l in range(self.num_layers)}
        
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
            
            print(f"Harvesting {i+1}/{len(problems)}...", end="\r")
            
        return layer_states

    def compute_steering_vector(self, faithful_data, corrected_data):
        print(f"\n{C.HEADER}PHASE 1: Harvesting Vectors...{C.ENDC}")
        
        print(f"  Harvesting FAITHFUL states...")
        faithful_states = self.get_hidden_states_at_error(faithful_data)
        
        print(f"\n  Harvesting CORRECTED states...")
        corrected_states = self.get_hidden_states_at_error(corrected_data)
        
        steering_vectors = {}
        print(f"\n  Computing Mean Difference (Faithful - Corrected)...")
        
        for l in range(self.num_layers):
            if not faithful_states[l] or not corrected_states[l]:
                continue
                
            f_stack = torch.stack(faithful_states[l])
            c_stack = torch.stack(corrected_states[l])
            
            f_mean = f_stack.mean(dim=0)
            c_mean = c_stack.mean(dim=0)
            
            vec = f_mean - c_mean
            
            steering_vectors[l] = vec.to(DEVICE)
            
        print(f"  Computed {len(steering_vectors)} steering vectors.")
        return steering_vectors

    def run_intervention(self, problems, vectors):
        print(f"\n{C.HEADER}PHASE 2: Steering Intervention (The Sedative)...{C.ENDC}")
        print(f"Injecting vector into Layers {min(INJECTION_LAYERS)}-{max(INJECTION_LAYERS)} with coeff {STEERING_COEFF}")
        
        results = []
        success_count = 0
        
        def get_steering_hook(layer_idx, vec):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    v = vec.to(dtype=hidden.dtype, device=hidden.device)
                    hidden = hidden + (v * STEERING_COEFF)
                    return (hidden,) + output[1:]
                else:
                    v = vec.to(dtype=output.dtype, device=output.device)
                    return output + (v * STEERING_COEFF)
            return hook

        hooks = []
        for l in INJECTION_LAYERS:
            if l in vectors:
                layer_module = self.model.model.layers[l]
                h = layer_module.register_forward_hook(get_steering_hook(l, vectors[l]))
                hooks.append(h)
                
        print(f"{C.WARNING}Steering active. {len(hooks)} hooks registered. Generating...{C.ENDC}\n")
        
        try:
            for i, prob in enumerate(problems):
                full_text = prob['continued_cot']
                inj_val = str(prob['injected_value'])
                idx = full_text.find(inj_val)
                if idx == -1: 
                    print(f"Skipping {i}: Injection not found in text")
                    continue
                
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
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_content = generated[len(base_prompt):] 
                
                final_ans = extract_answer(new_content)
                gt = prob['ground_truth_answer']
                
                is_faithful = True
                if final_ans is not None and gt is not None:
                    if abs(final_ans - gt) < 0.1:
                        is_faithful = False
                
                status = f"{C.OKGREEN}FORCED FAITHFUL{C.ENDC}" if is_faithful else f"{C.FAIL}STILL CORRECTED{C.ENDC}"
                print(f"[{i+1}/{len(problems)}] GT:{gt} | Gen:{final_ans} | {status}")
                
                if is_faithful: success_count += 1
                
                results.append({
                    "problem_id": prob['problem_id'],
                    "is_faithful": is_faithful,
                    "generated_text": new_content
                })
                
        finally:
            for h in hooks: h.remove()
            
        return success_count, len(results), results


def main():
    original_stdout = sys.stdout
    tee_stdout = TeeOutput(LOG_FILE, original_stdout)
    sys.stdout = tee_stdout
    
    try:
        print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {LOG_FILE}\n")
        
        print(f"Reading {INPUT_FILE}...")
        data = []
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
                
        faithful_set = [d for d in data if d.get('classification') is True]
        corrected_set = [d for d in data if d.get('classification') is False]
        
        print(f"Found {len(faithful_set)} Faithful, {len(corrected_set)} Corrected.")
        
        if len(faithful_set) < NUM_HARVEST_SAMPLES or len(corrected_set) < NUM_HARVEST_SAMPLES:
            print("Not enough data.")
            return

        exp = SteeringExperiment()
        
        f_harvest = faithful_set[:NUM_HARVEST_SAMPLES]
        c_harvest = corrected_set[:NUM_HARVEST_SAMPLES]
        
        vectors = exp.compute_steering_vector(f_harvest, c_harvest)
        
        test_set = corrected_set[NUM_HARVEST_SAMPLES:NUM_HARVEST_SAMPLES+NUM_TEST_SAMPLES]
        if len(test_set) < 10: test_set = corrected_set[:NUM_TEST_SAMPLES]
        
        success, total, results = exp.run_intervention(test_set, vectors)
        
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n{C.HEADER}{'='*60}{C.ENDC}")
        print(f"{C.HEADER}FINAL RESULTS{C.ENDC}")
        print(f"{C.HEADER}{'='*60}{C.ENDC}")
        print(f"Input Behavior: 100% Self-Correcting")
        print(f"After Steering: {success}/{total} became FAITHFUL")
        print(f"Success Rate:   {success/total:.1%}")
        print(f"Output saved to: {OUTPUT_FILE}")
        
        print(f"\n{C.OKGREEN}Analysis complete!{C.ENDC}")
        print(f"Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        sys.stdout = original_stdout
        tee_stdout.close()


if __name__ == "__main__":
    main()