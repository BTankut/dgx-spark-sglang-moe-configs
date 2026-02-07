#!/usr/bin/env python3
"""
A/B Benchmark: GLM-4.7-FP8 on SM121 (NVIDIA GB10)
Test A: Without GB10 MoE configs (fallback/default)
Test B: With GB10 MoE configs
"""
import subprocess, sys, time, json, os, shutil

CONFIG_DIR = os.path.expanduser('~/miniforge3/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs/')
BACKUP_DIR = os.path.expanduser('~/sm121-kernels/gb10_configs_backup/')
MODEL = 'zai-org/GLM-4.7-FP8'

# Benchmark parameters
PROMPTS = [
    'Explain quantum computing in simple terms.',
    'Write a Python function to calculate fibonacci numbers efficiently.',
    'What are the key differences between TCP and UDP protocols?',
    'Summarize the history of artificial intelligence in 200 words.',
]

def get_gb10_configs():
    return [f for f in os.listdir(CONFIG_DIR) if 'NVIDIA_GB10' in f]

def backup_configs():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for f in get_gb10_configs():
        shutil.copy2(os.path.join(CONFIG_DIR, f), os.path.join(BACKUP_DIR, f))

def remove_configs():
    for f in get_gb10_configs():
        os.remove(os.path.join(CONFIG_DIR, f))

def restore_configs():
    for f in os.listdir(BACKUP_DIR):
        shutil.copy2(os.path.join(BACKUP_DIR, f), os.path.join(CONFIG_DIR, f))

def run_benchmark(label, prompts):
    from vllm import LLM, SamplingParams
    
    print(f'\n=== {label} ===')
    print(f'Loading model: {MODEL}')
    
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # No CUDA graph for fair comparison
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    # Warmup
    print('Warmup...')
    llm.generate(['Hello'], sampling_params)
    
    # Benchmark
    results = []
    for i in range(3):  # 3 rounds
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_tokens / elapsed
        results.append({'round': i+1, 'tokens': total_tokens, 'time_s': elapsed, 'tok_per_s': tps})
        print(f'  Round {i+1}: {total_tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s')
    
    avg_tps = sum(r['tok_per_s'] for r in results) / len(results)
    print(f'  Average: {avg_tps:.1f} tok/s')
    
    # Cleanup GPU
    del llm
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return {'label': label, 'results': results, 'avg_tok_per_s': avg_tps}

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    
    if mode == 'without':
        # Test A: Remove GB10 configs
        backup_configs()
        remove_configs()
        print(f'GB10 configs: REMOVED ({len(os.listdir(BACKUP_DIR))} backed up)')
        result_a = run_benchmark('Test A: Without GB10 Configs (Fallback)', PROMPTS)
        restore_configs()
        print(json.dumps(result_a, indent=2))
    
    elif mode == 'with':
        # Test B: With GB10 configs
        configs = get_gb10_configs()
        print(f'GB10 configs: {len(configs)} present')
        result_b = run_benchmark('Test B: With GB10 Configs', PROMPTS)
        print(json.dumps(result_b, indent=2))
    
    elif mode == 'both':
        # Full A/B test
        # Test A first
        backup_configs()
        remove_configs()
        print(f'GB10 configs: REMOVED for Test A')
        result_a = run_benchmark('Test A: Without GB10 Configs (Fallback)', PROMPTS)
        restore_configs()
        
        time.sleep(5)
        
        # Test B
        configs = get_gb10_configs()
        print(f'GB10 configs: {len(configs)} present for Test B')
        result_b = run_benchmark('Test B: With GB10 Configs', PROMPTS)
        
        # Summary
        print('\n' + '='*60)
        print('SUMMARY')
        print('='*60)
        print(f'Test A (fallback): {result_a["avg_tok_per_s"]:.1f} tok/s')
        print(f'Test B (GB10 cfg): {result_b["avg_tok_per_s"]:.1f} tok/s')
        diff = result_b['avg_tok_per_s'] - result_a['avg_tok_per_s']
        pct = (diff / result_a['avg_tok_per_s']) * 100
        print(f'Difference: {diff:+.1f} tok/s ({pct:+.1f}%)')
        
        with open(os.path.expanduser('~/sm121-kernels/ab_results.json'), 'w') as f:
            json.dump({'test_a': result_a, 'test_b': result_b, 'diff_pct': pct}, f, indent=2)
        print(f'Results saved to ~/sm121-kernels/ab_results.json')
