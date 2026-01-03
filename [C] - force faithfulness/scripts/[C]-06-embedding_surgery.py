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

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "[A].jsonl"
OUTPUT_FILE = "[C]-06-output.json"
LOG_FILE = "[C]-06-logs.txt"
DEVICE = "mps"

STRENGTH = 30.0

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

class EmbeddingSurgeryExperiment:
    def __init__(self):
        print(f"{C.OKGREEN}Loading model: {MODEL_NAME} on {DEVICE}...{C.ENDC}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
        )
        self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Model loaded. Strategy: Embedding Surgery @ {STRENGTH}")

    def get_embeddings_at_error(self, problems):
        embeddings = []
        print(f"  Harvesting embeddings...")
        
        captured = {"val": None}
        def capture_hook(module, args, output):
            captured["val"] = output
            
        hook = self.model.model.embed_tokens.register_forward_hook(capture_hook)
        
        try:
            for i, prob in enumerate(problems):
                full_text = prob['continued_cot']
                inj_val = str(prob['injected_value'])
                idx = full_text.find(inj_val)
                if idx == -1: continue
                
                cutoff = idx + len(inj_val)
                prefix = full_text[:cutoff]
                inputs = self.tokenizer(prefix, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    self.model(**inputs, output_hidden_states=False)
                
                emb = captured["val"][0, -1, :].cpu()
                embeddings.append(emb)
        finally:
            hook.remove()
            
        return embeddings

    def compute_vector(self, faithful, corrected):
        print(f"\n{C.HEADER}PHASE 1: Computing Embedding Vector...{C.ENDC}")
        f_embs = self.get_embeddings_at_error(faithful)
        c_embs = self.get_embeddings_at_error(corrected)
        
        if not f_embs or not c_embs:
            print("Error: No embeddings captured.")
            return None
            
        f_stack = torch.stack(f_embs).float()
        c_stack = torch.stack(c_embs).float()
        
        vec = f_stack.mean(dim=0) - c_stack.mean(dim=0)
        norm = torch.norm(vec)
        if norm > 1e-6: vec = vec / norm
        
        return vec.to(dtype=self.model.dtype, device=DEVICE)

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

    def run_surgery(self, problems, vector):
        print(f"\n{C.HEADER}PHASE 2: The Trojan Horse (Embedding Injection)...{C.ENDC}")
        
        class State:
            target_indices = None
        state = State()

        def get_surgery_hook(vec):
            def hook(module, args, output):
                embeds = output
                seq_len = embeds.shape[1]
                
                if seq_len > 1 and state.target_indices:
                    v = vec.to(dtype=embeds.dtype, device=embeds.device)
                    for b, indices in enumerate(state.target_indices):
                        for idx in indices:
                            if idx < seq_len:
                                embeds[b, idx, :] += (v * STRENGTH)
                
                return embeds
            return hook

        h = self.model.model.embed_tokens.register_forward_hook(get_surgery_hook(vector))

        results = []
        stats = {"faithful": 0, "corrected": 0, "broken": 0}
        
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
                        **inputs, max_new_tokens=1024, do_sample=False, 
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                part = gen_text.split(context_prefix)[-1] if context_prefix in gen_text else gen_text[-200:]
                final_ans = extract_answer(part)
                gt = prob['ground_truth_answer']
                
                is_valid = True
                if len(part) < 5 or len(set(part[-50:])) < 8: is_valid = False
                
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
                        status = f"{C.OKGREEN}FAITHFUL{C.ENDC}"
                else:
                    stats["broken"] += 1
                    status = f"{C.WARNING}NO_ANS{C.ENDC}"

                print(f"[{i+1}/50] GT:{gt} | Gen:{final_ans} | {status}")
                results.append(part)

        finally:
            h.remove()
            
        return stats, results

def main():
    sys.stdout = TeeOutput(LOG_FILE, sys.stdout)
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    faithful = [d for d in data if d.get('classification') is True]
    corrected = [d for d in data if d.get('classification') is False]
    
    exp = EmbeddingSurgeryExperiment()
    vector = exp.compute_vector(faithful[:50], corrected[:50])
    if vector is not None:
        stats, _ = exp.run_surgery(corrected[:50], vector)
        
        print(f"\nRESULTS:")
        print(f"Faithful: {stats['faithful']} ({stats['faithful']/50:.1%})")
        print(f"Corrected: {stats['corrected']} ({stats['corrected']/50:.1%})")
        print(f"Broken:    {stats['broken']} ({stats['broken']/50:.1%})")

if __name__ == "__main__":
    main()