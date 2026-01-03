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
OUTPUT_FILE = "[C]-01-output.json"
LOG_FILE = "[C]-01-logs.txt"
DEVICE = "mps"

TOP_K_HEADS = 20
NUM_DETECTION_SAMPLES = 20
NUM_INTERVENTION_SAMPLES = 50

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
    patterns = [
        r'boxed\{([\d\.]+)\}',
        r'is\s+([\d\.]+)\.',
        r'=\s*([\d\.]+)',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m: return float(m.group(1))
    return None


class HeadAblationExperiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
            attn_implementation="eager"
        ).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        
        print(f"Model loaded: {self.num_layers} layers, {self.num_heads} heads.")

    def detect_reset_heads(self, problems):
        print(f"\n{C.HEADER}PHASE 1: Detecting Reset Heads (Scanning {len(problems)} samples)...{C.ENDC}")
        
        bos_scores = torch.zeros((self.num_layers, self.num_heads)).to(DEVICE)
        
        for i, prob in enumerate(problems):
            full_text = prob['continued_cot']
            inputs = self.tokenizer(full_text, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            seq_len = inputs.input_ids.shape[1]
            start_idx = max(0, seq_len - 50) 
            
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                attn_to_bos = layer_attn[0, :, start_idx:, 0] 
                avg_score = attn_to_bos.mean(dim=1)
                bos_scores[layer_idx] += avg_score
            
            print(f"Scanning {i+1}/{len(problems)}...", end="\r")

        bos_scores /= len(problems)
        
        flat_indices = torch.topk(bos_scores.flatten(), TOP_K_HEADS).indices
        
        heads_to_ablate = []
        print(f"\n\n{C.BOLD}Top {TOP_K_HEADS} Reset Heads identified:{C.ENDC}")
        for idx in flat_indices:
            l = (idx // self.num_heads).item()
            h = (idx % self.num_heads).item()
            score = bos_scores[l, h].item()
            heads_to_ablate.append((l, h))
            print(f"  Layer {l}, Head {h} (Score: {score:.4f})")
            
        return heads_to_ablate

    def run_intervention(self, problems, heads_to_ablate):
        print(f"\n{C.HEADER}PHASE 2: Lobotomy (Intervention on {len(problems)} samples)...{C.ENDC}")
        
        hooks = []
        
        def get_ablation_hook(heads_in_layer):
            def hook(module, args, output):
                hidden = output[0]
                
                for h in heads_in_layer:
                    start = h * self.head_dim
                    end = (h + 1) * self.head_dim
                    hidden[:, :, start:end] = 0.0
                
                return (hidden,) + output[1:]
            return hook

        layer_map = {}
        for l, h in heads_to_ablate:
            if l not in layer_map: layer_map[l] = []
            layer_map[l].append(h)
            
        for l, heads in layer_map.items():
            layer_module = self.model.model.layers[l].self_attn
            h = layer_module.register_forward_hook(get_ablation_hook(heads))
            hooks.append(h)
            
        print(f"{C.WARNING}Hooks registered. Generating...{C.ENDC}\n")
        
        results = []
        forced_faithfulness_count = 0
        
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
                
                if is_faithful: forced_faithfulness_count += 1
                
                results.append({
                    "problem_id": prob['problem_id'],
                    "is_faithful_after_ablation": is_faithful,
                    "generated_text": new_content
                })
                
        finally:
            for h in hooks: h.remove()
            
        return forced_faithfulness_count, len(results), results


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
                
        corrected_problems = [d for d in data if d.get('classification') is False]
        
        print(f"Found {len(corrected_problems)} Self-Corrected examples.")
        if len(corrected_problems) < 10:
            print("Not enough examples to run.")
            return

        exp = HeadAblationExperiment()
        
        detect_set = corrected_problems[:NUM_DETECTION_SAMPLES]
        reset_heads = exp.detect_reset_heads(detect_set)
        
        intervention_set = corrected_problems[NUM_DETECTION_SAMPLES:NUM_DETECTION_SAMPLES+NUM_INTERVENTION_SAMPLES]
        if len(intervention_set) < 10: intervention_set = corrected_problems[:NUM_INTERVENTION_SAMPLES]
        
        success_count, total, results = exp.run_intervention(intervention_set, reset_heads)
        
        print(f"\n{C.OKGREEN}Saving results to {OUTPUT_FILE}...{C.ENDC}")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n{C.HEADER}{'='*60}{C.ENDC}")
        print(f"{C.HEADER}FINAL RESULTS{C.ENDC}")
        print(f"{C.HEADER}{'='*60}{C.ENDC}")
        print(f"Original Behavior: 100% Self-Correcting")
        print(f"After Lobotomy:    {success_count}/{total} became FAITHFUL")
        print(f"Success Rate:      {success_count/total:.1%}")
        print(f"Output saved to:   {OUTPUT_FILE}")
        
        print(f"\n{C.OKGREEN}Analysis complete!{C.ENDC}")
        print(f"Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        sys.stdout = original_stdout
        tee_stdout.close()


if __name__ == "__main__":
    main()