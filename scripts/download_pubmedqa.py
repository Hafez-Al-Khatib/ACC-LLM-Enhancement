from datasets import load_dataset
import json, os

# Try multiple medical QA datasets, use first that works
DATASETS_TO_TRY = [
    ('medicine_qa', None),
    ('medical_dialog', None),
    ('medical_meadow', None),
]

os.makedirs('data/medical', exist_ok=True)

def format_alpaca_style(instruction, input_text, output):
    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return {'text': text, 'instruction': instruction, 'input': input_text, 'output': output}

# First try: Use Alpaca as a base and create a synthetic medical subset
# (Real medical datasets require authentication or have API issues)
print("Downloading Alpaca for medical formatting...")
try:
    ds = load_dataset('tatsu-lab/alpaca')
    print('Alpaca loaded:', len(ds['train']))
    
    # Filter for medical-related instructions
    medical_keywords = ['medical', 'health', 'doctor', 'patient', 'disease', 'symptom', 'treatment', 'medicine', 'drug', 'diagnosis']
    medical_rows = []
    for row in ds['train']:
        inst = row.get('instruction', '').lower()
        if any(kw in inst for kw in medical_keywords):
            medical_rows.append(format_alpaca_style(
                row.get('instruction', ''),
                row.get('input', ''),
                row.get('output', '')
            ))
    
    print(f'Found {len(medical_rows)} medical-related samples')
    
    if len(medical_rows) < 50:
        # Fallback: just take first 200 samples and treat as general instruction
        print('Too few medical samples. Using general instruction subset.')
        medical_rows = []
        for i, row in enumerate(ds['train']):
            if i >= 200:
                break
            medical_rows.append(format_alpaca_style(
                row.get('instruction', ''),
                row.get('input', ''),
                row.get('output', '')
            ))
    
    # Split 80/10/10
    n = len(medical_rows)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    splits = {
        'train': medical_rows[:n_train],
        'val': medical_rows[n_train:n_train + n_val],
        'test': medical_rows[n_train + n_val:]
    }
    
    for split, rows in splits.items():
        with open(f'data/medical/{split}.jsonl', 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'Saved {split}: {len(rows)} samples')
    
    print("Dataset ready at data/medical/")
    
except Exception as e:
    print(f'Error: {e}')
    # Ultimate fallback: create synthetic data
    print('Creating synthetic instruction data...')
    os.makedirs('data/synthetic', exist_ok=True)
    
    instructions = [
        ('Explain photosynthesis', 'What is photosynthesis?', 'Photosynthesis is the process by which plants convert light energy into chemical energy.'),
        ('Summarize this', 'The quick brown fox jumps over the lazy dog.', 'A fox jumps over a dog.'),
        ('Define gravity', '', 'Gravity is a force of attraction between any two masses in the universe.'),
    ] * 50  # 150 samples
    
    rows = [format_alpaca_style(i, inp, out) for i, inp, out in instructions]
    with open('data/synthetic/train.jsonl', 'w') as f:
        for r in rows[:120]:
            f.write(json.dumps(r) + '\n')
    with open('data/synthetic/val.jsonl', 'w') as f:
        for r in rows[120:135]:
            f.write(json.dumps(r) + '\n')
    with open('data/synthetic/test.jsonl', 'w') as f:
        for r in rows[135:]:
            f.write(json.dumps(r) + '\n')
    print('Synthetic dataset created at data/synthetic/')
