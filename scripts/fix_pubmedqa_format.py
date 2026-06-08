#!/usr/bin/env python3
"""Fix PubMedQA dataset formatting by extracting clean context text from raw dict strings."""

import json
import glob
import re


def clean_record(record):
    input_text = record.get('input', '')
    if "Context: {'contexts':" not in input_text:
        return record
    
    match = re.search(r"Context: (\{.*\})\nQuestion:", input_text, re.DOTALL)
    if not match:
        return record
    
    try:
        ctx_dict = eval(match.group(1))
        contexts = ctx_dict.get('contexts', [])
        if contexts:
            ctx_text = '\n'.join(contexts)
        else:
            ctx_text = str(ctx_dict)
        
        question = input_text.split('\nQuestion: ')[-1]
        record['input'] = f'Context: {ctx_text}\nQuestion: {question}'
        record['text'] = (
            f"### Instruction:\n{record['instruction']}\n\n"
            f"### Input:\n{record['input']}\n\n"
            f"### Response:\n{record['output']}"
        )
    except Exception as e:
        print(f"  Skip (eval failed): {e}")
    
    return record


def main():
    for path in glob.glob('experiments/datasets/pubmedqa/*.jsonl'):
        print(f"Cleaning {path}...")
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(clean_record(json.loads(line)))
        with open(path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"  Wrote {len(records)} records")


if __name__ == "__main__":
    main()
