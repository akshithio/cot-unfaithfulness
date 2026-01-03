import json
import numpy as np
import re
from scipy import stats

# Load data
problems = []
with open('[A].jsonl', 'r') as f:
    for line in f:
        problems.append(json.loads(line))

faithful = [p for p in problems if p['classification'] == True]
corrected = [p for p in problems if p['classification'] == False]

print(f"Dataset: {len(faithful)} faithful, {len(corrected)} corrected\n")
print("="*80)

# ============================================================================
# ANALYSIS 1: Error Magnitude
# ============================================================================
print("\n1. ERROR MAGNITUDE EFFECT")
print("-" * 40)

for p in problems:
    try:
        p['error_magnitude'] = abs(float(p['injected_value']) - float(p['original_value']))
    except:
        p['error_magnitude'] = None

faithful_mags = [p['error_magnitude'] for p in faithful if p['error_magnitude'] is not None]
corrected_mags = [p['error_magnitude'] for p in corrected if p['error_magnitude'] is not None]

print(f"Faithful:    mean = {np.mean(faithful_mags):.2f}, median = {np.median(faithful_mags):.2f}")
print(f"Corrected:   mean = {np.mean(corrected_mags):.2f}, median = {np.median(corrected_mags):.2f}")
t_stat, p_val = stats.ttest_ind(faithful_mags, corrected_mags)
print(f"T-test: t = {t_stat:.3f}, p = {p_val:.4f}")

# ============================================================================
# ANALYSIS 2: Error Position (Relative)
# ============================================================================
print("\n2. ERROR POSITION IN REASONING CHAIN")
print("-" * 40)

for p in problems:
    cot = p['continued_cot']
    injected = str(p['injected_value'])
    pos = cot.find(injected)
    if pos != -1:
        p['relative_position'] = pos / len(cot)
    else:
        p['relative_position'] = None

faithful_pos = [p['relative_position'] for p in faithful if p['relative_position'] is not None]
corrected_pos = [p['relative_position'] for p in corrected if p['relative_position'] is not None]

print(f"Faithful:    mean = {np.mean(faithful_pos):.3f}, median = {np.median(faithful_pos):.3f}")
print(f"Corrected:   mean = {np.mean(corrected_pos):.3f}, median = {np.median(corrected_pos):.3f}")
print(f"(0.0 = start, 1.0 = end)")
t_stat, p_val = stats.ttest_ind(faithful_pos, corrected_pos)
print(f"T-test: t = {t_stat:.3f}, p = {p_val:.4f}")

# ============================================================================
# ANALYSIS 3: Problem Complexity (Number of Operations)
# ============================================================================
print("\n3. PROBLEM COMPLEXITY")
print("-" * 40)

for p in problems:
    # Count mathematical operations in problem text
    ops = len(re.findall(r'\d+\s*[\+\-\*\/×÷]\s*\d+', p['problem_text']))
    # Also count numbers as proxy for complexity
    nums = len(re.findall(r'\d+', p['problem_text']))
    p['num_operations'] = ops
    p['num_numbers'] = nums

faithful_ops = [p['num_operations'] for p in faithful]
corrected_ops = [p['num_operations'] for p in corrected]

print(f"Operations in problem text:")
print(f"  Faithful:    mean = {np.mean(faithful_ops):.2f}")
print(f"  Corrected:   mean = {np.mean(corrected_ops):.2f}")
t_stat, p_val = stats.ttest_ind(faithful_ops, corrected_ops)
print(f"  T-test: t = {t_stat:.3f}, p = {p_val:.4f}")

faithful_nums = [p['num_numbers'] for p in faithful]
corrected_nums = [p['num_numbers'] for p in corrected]

print(f"\nNumbers in problem text:")
print(f"  Faithful:    mean = {np.mean(faithful_nums):.2f}")
print(f"  Corrected:   mean = {np.mean(corrected_nums):.2f}")
t_stat, p_val = stats.ttest_ind(faithful_nums, corrected_nums)
print(f"  T-test: t = {t_stat:.3f}, p = {p_val:.4f}")

# ============================================================================
# ANALYSIS 4: Error Impact on Final Answer
# ============================================================================
print("\n4. ERROR IMPACT ON FINAL ANSWER")
print("-" * 40)

for p in problems:
    try:
        ground_truth = float(p['ground_truth_answer'])
        final = float(p['final_answer'])
        p['answer_error'] = abs(final - ground_truth)
        p['relative_answer_error'] = p['answer_error'] / max(ground_truth, 1.0)
    except:
        p['answer_error'] = None
        p['relative_answer_error'] = None

faithful_err = [p['relative_answer_error'] for p in faithful if p['relative_answer_error'] is not None]
corrected_err = [p['relative_answer_error'] for p in corrected if p['relative_answer_error'] is not None]

print(f"Relative error in final answer:")
print(f"  Faithful:    mean = {np.mean(faithful_err):.3f}")
print(f"  Corrected:   mean = {np.mean(corrected_err):.3f}")
print(f"  (This should differ by design - faithful propagate errors)")

# ============================================================================
# ANALYSIS 5: Injection Point Characteristics
# ============================================================================
print("\n5. INJECTION POINT CONTEXT")
print("-" * 40)

for p in problems:
    inj_point = p.get('injection_point', '')
    # Check if injection is in a calculation vs just stating a value
    has_equals = '=' in inj_point
    has_frac = 'frac' in inj_point or '/' in inj_point
    p['injection_is_calculation'] = has_equals
    p['injection_is_fraction'] = has_frac

faithful_calc = sum(p['injection_is_calculation'] for p in faithful) / len(faithful)
corrected_calc = sum(p['injection_is_calculation'] for p in corrected) / len(corrected)

print(f"Error injected into calculation (has '='):")
print(f"  Faithful:    {faithful_calc:.1%}")
print(f"  Corrected:   {corrected_calc:.1%}")

# ============================================================================
# ANALYSIS 6: Value Magnitude (absolute, not error magnitude)
# ============================================================================
print("\n6. ABSOLUTE VALUE MAGNITUDES")
print("-" * 40)

for p in problems:
    try:
        p['original_value_mag'] = abs(float(p['original_value']))
        p['injected_value_mag'] = abs(float(p['injected_value']))
    except:
        p['original_value_mag'] = None
        p['injected_value_mag'] = None

faithful_orig = [p['original_value_mag'] for p in faithful if p['original_value_mag'] is not None]
corrected_orig = [p['original_value_mag'] for p in corrected if p['original_value_mag'] is not None]

print(f"Original value magnitude:")
print(f"  Faithful:    mean = {np.mean(faithful_orig):.2f}, median = {np.median(faithful_orig):.2f}")
print(f"  Corrected:   mean = {np.mean(corrected_orig):.2f}, median = {np.median(corrected_orig):.2f}")
t_stat, p_val = stats.ttest_ind(faithful_orig, corrected_orig)
print(f"  T-test: t = {t_stat:.3f}, p = {p_val:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: What predicts self-correction?")
print("="*80)

# Collect all p-values
tests = {
    'Error magnitude': stats.ttest_ind(faithful_mags, corrected_mags)[1],
    'Error position': stats.ttest_ind(faithful_pos, corrected_pos)[1],
    'Problem complexity (ops)': stats.ttest_ind(faithful_ops, corrected_ops)[1],
    'Problem complexity (nums)': stats.ttest_ind(faithful_nums, corrected_nums)[1],
    'Original value magnitude': stats.ttest_ind(faithful_orig, corrected_orig)[1],
}

print("\nSignificant predictors (p < 0.05):")
for name, p in sorted(tests.items(), key=lambda x: x[1]):
    if p < 0.05:
        print(f"  ✓ {name:30s} p = {p:.4f}")
    else:
        print(f"  ✗ {name:30s} p = {p:.4f}")