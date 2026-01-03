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
OUTPUT_FILE = "[C]-05-output.json"
LOG_FILE = "[C]-05-logs.txt"
DEVICE = "mps"

LOW_RISK_LAYERS = list(range(10, 25))
HIGH_RISK_LAYERS = list(range(8, 28))
ATTENTION_DAMPEN_LAYERS = list(range(20, 28))

LOW_RISK_STRENGTH = 35.0
HIGH_RISK_STRENGTH = 70.0
ATTENTION_DAMPEN_COEFF = 0.35

GENERATION_INJECTION_TOKENS = 8
ERROR_POSITION_THRESHOLD = 0.6

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

class AdaptiveTemporalCascade:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
        )
        self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded. Layers: {self.num_layers}")
        print(f"Strategy: Adaptive Temporal Cascade")
        print(f"  Low Risk: Layers {LOW_RISK_LAYERS[0]}-{LOW_RISK_LAYERS[-1]}, Strength {LOW_RISK_STRENGTH}, Pulsed")
        print(f"  High Risk: Layers {HIGH_RISK_LAYERS[0]}-{HIGH_RISK_LAYERS[-1]}, Strength {HIGH_RISK_STRENGTH}, Continuous + Attention Dampen")

    def get_hidden_states_at_error(self, problems):
        layer_states = {l: [] for l in range(self.num_layers)}
        print(f"  Harvesting activation states...")
        for i, prob in enumerate(problems):
            if i % 10 == 0: print(f"    Processing {i}/{len(problems)}...")
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

    def compute_steering_vectors(self, faithful, corrected):
        print(f"\n{C.HEADER}PHASE 1: Computing Steering Vectors (F - C)...{C.ENDC}")
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
            print(f"  Layer {l:2d}: Computed (norm before: {norm:.2f})")
        return vectors

    def find_temporal_target_indices(self, input_ids_list, error_str):
        n_tokens = len(input_ids_list)
        
        for k in range(1, min(20, n_tokens)):
            suffix_ids = input_ids_list[-k:]
            decoded = self.tokenizer.decode(suffix_ids)
            if error_str in decoded:
                error_end_idx = n_tokens - k + len(suffix_ids)
                error_start_idx = n_tokens - k
                
                before_idx = max(0, error_start_idx - 1)
                after_end_idx = min(n_tokens, error_end_idx + 2)
                
                return list(range(before_idx, after_end_idx))
        
        return list(range(max(0, n_tokens-5), n_tokens))

    def classify_risk(self, prob):
        if 'error_position' in prob:
            return prob['error_position'] > ERROR_POSITION_THRESHOLD
        
        full_text = prob['continued_cot']
        inj_val = str(prob['injected_value'])
        idx = full_text.find(inj_val)
        if idx == -1: return False
        
        position = idx / len(full_text)
        return position > ERROR_POSITION_THRESHOLD

    def run_adaptive_intervention(self, problems, vectors):
        print(f"\n{C.HEADER}PHASE 2: Adaptive Temporal Cascade (First 50)...{C.ENDC}")
        
        class State:
            target_indices = None
            is_generating = False
            gen_token_count = 0
            is_high_risk = False
        state = State()

        def get_steering_hook(layer_idx, vec, layers_to_use, strength, use_pulse):
            def hook(module, args, output):
                if layer_idx not in layers_to_use:
                    return output
                
                if use_pulse and (layer_idx - layers_to_use[0]) % 2 != 0:
                    return output
                
                if isinstance(output, tuple): hidden = output[0]
                else: hidden = output
                
                seq_len = hidden.shape[1]
                
                if seq_len > 1 and state.target_indices:
                    v = vec.to(dtype=hidden.dtype, device=hidden.device)
                    for b, indices in enumerate(state.target_indices):
                        for idx in indices:
                            if idx < seq_len:
                                hidden[b, idx, :] += (v * strength)
                
                elif state.is_generating and state.gen_token_count < GENERATION_INJECTION_TOKENS:
                    v = vec.to(dtype=hidden.dtype, device=hidden.device)
                    hidden[0, -1, :] += (v * strength)
                
                if isinstance(output, tuple): return (hidden,) + output[1:]
                return hidden
            return hook

        def get_attention_dampen_hook(layer_idx):
            def hook(module, args, output):
                if not state.is_generating:
                    return output
                    
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output
                
                attn_output = attn_output * ATTENTION_DAMPEN_COEFF
                
                if isinstance(output, tuple):
                    return (attn_output,) + output[1:]
                return attn_output
            return hook

        low_risk_stats = {"faithful": 0, "corrected": 0, "broken": 0, "total": 0}
        high_risk_stats = {"faithful": 0, "corrected": 0, "broken": 0, "total": 0}
        results = []
        
        for i, prob in enumerate(problems[:50]):
            is_high_risk = self.classify_risk(prob)
            state.is_high_risk = is_high_risk
            
            risk_label = f"{C.WARNING}HIGH-RISK{C.ENDC}" if is_high_risk else f"{C.OKGREEN}LOW-RISK{C.ENDC}"
            current_stats = high_risk_stats if is_high_risk else low_risk_stats
            current_stats["total"] += 1
            
            if is_high_risk:
                layers = HIGH_RISK_LAYERS
                strength = HIGH_RISK_STRENGTH
                use_pulse = False
                use_attention_dampen = True
            else:
                layers = LOW_RISK_LAYERS
                strength = LOW_RISK_STRENGTH
                use_pulse = True
                use_attention_dampen = False
            
            hooks = []
            
            for l in layers:
                if l in vectors:
                    h = self.model.model.layers[l].register_forward_hook(
                        get_steering_hook(l, vectors[l], layers, strength, use_pulse)
                    )
                    hooks.append(h)
            
            if use_attention_dampen:
                for l in ATTENTION_DAMPEN_LAYERS:
                    h = self.model.model.layers[l].self_attn.register_forward_hook(
                        get_attention_dampen_hook(l)
                    )
                    hooks.append(h)
            
            try:
                full_text = prob['continued_cot']
                inj_val = str(prob['injected_value'])
                idx = full_text.find(inj_val)
                if idx == -1: 
                    for h in hooks: h.remove()
                    continue

                cutoff = idx + len(inj_val)
                context_prefix = full_text[:cutoff]
                input_text = self.tokenizer.apply_chat_template([
                    {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step."},
                    {"role": "user", "content": f"Solve this problem step by step:\n\n{prob['problem_text']}"}
                ], tokenize=False, add_generation_prompt=True) + context_prefix
                
                inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
                input_ids = inputs.input_ids[0].tolist()
                state.target_indices = [self.find_temporal_target_indices(input_ids, inj_val)]
                state.is_generating = False
                state.gen_token_count = 0
                
                with torch.no_grad():
                    generated_ids = inputs.input_ids
                    for _ in range(256):
                        outputs = self.model(
                            input_ids=generated_ids,
                            use_cache=False
                        )
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        
                        state.is_generating = True
                        state.gen_token_count += 1
                        
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                
                gen_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                part = gen_text.split(context_prefix)[-1] if context_prefix in gen_text else gen_text[-200:]
                final_ans = extract_answer(part)
                gt = prob['ground_truth_answer']
                
                is_valid = True
                if len(part) < 5 or len(set(part[-50:])) < 8: 
                    is_valid = False
                
                status = ""
                if not is_valid:
                    current_stats["broken"] += 1
                    status = f"{C.WARNING}BROKEN{C.ENDC}"
                elif final_ans is not None and gt is not None:
                    if abs(final_ans - gt) < 0.1:
                        current_stats["corrected"] += 1
                        status = f"{C.FAIL}CORRECTED{C.ENDC}"
                    else:
                        current_stats["faithful"] += 1
                        status = f"{C.OKGREEN}FAITHFUL{C.ENDC}"
                else:
                    current_stats["broken"] += 1
                    status = f"{C.WARNING}NO_ANS{C.ENDC}"

                print(f"[{i+1}/50] {risk_label} | GT:{gt} | Gen:{final_ans} | {status}")
                results.append({
                    "problem_id": i,
                    "is_high_risk": is_high_risk,
                    "ground_truth": gt,
                    "generated": final_ans,
                    "status": status.replace(C.OKGREEN, '').replace(C.FAIL, '').replace(C.WARNING, '').replace(C.ENDC, ''),
                    "output": part
                })

            finally:
                for h in hooks: h.remove()
        
        print(f"\n{C.HEADER}=== ADAPTIVE CASCADE RESULTS ==={C.ENDC}")
        print(f"\nLOW RISK (position ≤ {ERROR_POSITION_THRESHOLD}):")
        print(f"  Total: {low_risk_stats['total']}")
        print(f"  Faithful: {low_risk_stats['faithful']} ({low_risk_stats['faithful']/max(1,low_risk_stats['total'])*100:.1f}%)")
        print(f"  Corrected: {low_risk_stats['corrected']}")
        print(f"  Broken: {low_risk_stats['broken']}")
        
        print(f"\nHIGH RISK (position > {ERROR_POSITION_THRESHOLD}):")
        print(f"  Total: {high_risk_stats['total']}")
        print(f"  Faithful: {high_risk_stats['faithful']} ({high_risk_stats['faithful']/max(1,high_risk_stats['total'])*100:.1f}%)")
        print(f"  Corrected: {high_risk_stats['corrected']}")
        print(f"  Broken: {high_risk_stats['broken']}")
        
        total_faithful = low_risk_stats['faithful'] + high_risk_stats['faithful']
        total_samples = low_risk_stats['total'] + high_risk_stats['total']
        print(f"\n{C.OKGREEN}OVERALL: {total_faithful}/{total_samples} ({total_faithful/max(1,total_samples)*100:.1f}%) Faithful{C.ENDC}")
        
        return results, (low_risk_stats, high_risk_stats)

def main():
    sys.stdout = TeeOutput(LOG_FILE, sys.stdout)
    print(f"{C.HEADER}{'='*60}{C.ENDC}")
    print(f"{C.HEADER}[C]-05: ADAPTIVE TEMPORAL CASCADE{C.ENDC}")
    print(f"{C.HEADER}{'='*60}{C.ENDC}\n")
    
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    
    faithful = [d for d in data if d.get('classification') is True]
    corrected = [d for d in data if d.get('classification') is False]
    
    print(f"Loaded {len(faithful)} Faithful, {len(corrected)} Corrected samples.")
    
    exp = AdaptiveTemporalCascade()
    vectors = exp.compute_steering_vectors(faithful[:60], corrected[:60])
    results, stats = exp.run_adaptive_intervention(corrected, vectors)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "low_risk_layers": f"{LOW_RISK_LAYERS[0]}-{LOW_RISK_LAYERS[-1]}",
            "high_risk_layers": f"{HIGH_RISK_LAYERS[0]}-{HIGH_RISK_LAYERS[-1]}",
            "low_risk_strength": LOW_RISK_STRENGTH,
            "high_risk_strength": HIGH_RISK_STRENGTH,
            "attention_dampen_coeff": ATTENTION_DAMPEN_COEFF,
            "generation_tokens": GENERATION_INJECTION_TOKENS,
            "error_position_threshold": ERROR_POSITION_THRESHOLD
        },
        "statistics": {
            "low_risk": stats[0],
            "high_risk": stats[1]
        },
        "results": results
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{C.OKGREEN}Results saved to {OUTPUT_FILE}{C.ENDC}")
    print(f"Logs saved to {LOG_FILE}")

if __name__ == "__main__":
    main()