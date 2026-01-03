import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import sys

class C:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[C]-04-output.json"
LOG_FILE = "[C]-04-logs.txt"
DEVICE = "mps"

INJECTION_LAYERS = range(12, 25) 
STEERING_COEFF = 20.0
TEST_SAMPLES = 40

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

class DirectionalityExperiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
        )
        self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded. Comparing TARGETED vs RANDOM vs INVERSE.")

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
            
        return vectors

    def find_target_indices(self, input_ids_list, error_str):
        n_tokens = len(input_ids_list)
        for k in range(1, 15):
            suffix_ids = input_ids_list[-k:]
            decoded = self.tokenizer.decode(suffix_ids)
            if error_str in decoded:
                start_idx = n_tokens - k
                buffer_idx = max(0, start_idx - 1)
                return list(range(buffer_idx, n_tokens))
        return [n_tokens-1, n_tokens-2, n_tokens-3]

    def run_condition(self, condition_name, problems, vectors):
        print(f"\n{C.HEADER}RUNNING CONDITION: {condition_name}{C.ENDC}")
        
        active_vectors = {}
        for l, vec in vectors.items():
            if condition_name == "TARGETED":
                active_vectors[l] = vec
            elif condition_name == "INVERSE":
                active_vectors[l] = -1 * vec
            elif condition_name == "RANDOM":
                rand_v = torch.randn_like(vec)
                rand_v = rand_v / torch.norm(rand_v)
                active_vectors[l] = rand_v
        
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
                    for b, indices in enumerate(state.target_indices):
                        for idx in indices:
                            if idx < seq_len:
                                hidden[b, idx, :] += (v * STEERING_COEFF)
                
                if isinstance(output, tuple): return (hidden,) + output[1:]
                return hidden
            return hook

        hooks = []
        for l in INJECTION_LAYERS:
            if l in active_vectors:
                h = self.model.model.layers[l].register_forward_hook(get_hook(l, active_vectors[l]))
                hooks.append(h)

        success = 0
        broken = 0
        
        try:
            for i, prob in enumerate(problems):
                full_text = prob['continued_cot']
                inj_val = str(prob['injected_value'])
                idx = full_text.find(inj_val)
                if idx == -1: continue

                cutoff = idx + len(inj_val)
                context_prefix = full_text[:cutoff]
                
                input_text = self.tokenizer.apply_chat_template([
                    {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                    {"role": "user", "content": f"Solve this problem step by step:\n\n{prob['problem_text']}"}
                ], tokenize=False, add_generation_prompt=True) + context_prefix
                
                inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
                input_ids = inputs.input_ids[0].tolist()
                
                state.target_indices = [self.find_target_indices(input_ids, inj_val)]
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=128, do_sample=False, 
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                part = generated_full.split(context_prefix)[-1] if context_prefix in generated_full else generated_full[-200:]
                
                final_ans = extract_answer(part)
                gt = prob['ground_truth_answer']
                
                is_faithful = False
                if len(part) < 10 or len(set(part[-50:])) < 10:
                    broken += 1
                elif final_ans is not None and gt is not None:
                    if abs(final_ans - gt) > 0.1:
                        is_faithful = True
                        success += 1
                
                print(f"[{i+1}] {condition_name} | GT:{gt} | Gen:{final_ans} | Faith:{is_faithful}")
                
        finally:
            for h in hooks: h.remove()
            
        return success, broken

def main():
    sys.stdout = TeeOutput(LOG_FILE, sys.stdout)
    
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
            
    faithful = [d for d in data if d.get('classification') is True]
    corrected = [d for d in data if d.get('classification') is False]
    
    exp = DirectionalityExperiment()
    vectors = exp.compute_vectors(faithful[:40], corrected[:40])
    test_set = corrected[:TEST_SAMPLES]
    
    print(f"\n{C.BOLD}STARTING CONTROL EXPERIMENT (N={TEST_SAMPLES}){C.ENDC}")
    
    s_target, b_target = exp.run_condition("TARGETED", test_set, vectors)
    s_random, b_random = exp.run_condition("RANDOM", test_set, vectors)
    s_inverse, b_inverse = exp.run_condition("INVERSE", test_set, vectors)
    
    print(f"\n{C.HEADER}FINAL CONTROL RESULTS{C.ENDC}")
    print(f"TARGETED: {s_target}/{TEST_SAMPLES} ({s_target/TEST_SAMPLES:.1%}) | Broken: {b_target}")
    print(f"RANDOM:   {s_random}/{TEST_SAMPLES} ({s_random/TEST_SAMPLES:.1%}) | Broken: {b_random}")
    print(f"INVERSE:  {s_inverse}/{TEST_SAMPLES} ({s_inverse/TEST_SAMPLES:.1%}) | Broken: {b_inverse}")


if __name__ == "__main__":
    main()