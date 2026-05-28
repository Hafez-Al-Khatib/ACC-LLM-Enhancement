"""Generate synthetic training data for Approach B conflict detector."""

import json
import os

# Synthetic (prompt, label) pairs for training the conflict detector
# Labels: supported, hallucinated, uncertain, contradictory

TRAIN_DATA = [
    # Supported - factual, verifiable prompts
    ("The capital of France is Paris.", "supported"),
    ("Water freezes at 0 degrees Celsius.", "supported"),
    ("The Earth orbits the Sun.", "supported"),
    ("Humans have 23 pairs of chromosomes.", "supported"),
    ("Oxygen is essential for human respiration.", "supported"),
    ("The speed of light is approximately 300,000 km/s.", "supported"),
    ("DNA stands for deoxyribonucleic acid.", "supported"),
    ("The heart pumps blood through the circulatory system.", "supported"),
    ("Newton's first law describes inertia.", "supported"),
    ("Photosynthesis converts CO2 and water into glucose.", "supported"),
    
    # Hallucinated - plausible but false
    ("The Great Wall of China is visible from the moon.", "hallucinated"),
    ("Humans only use 10% of their brain capacity.", "hallucinated"),
    ("Goldfish have a 3-second memory.", "hallucinated"),
    ("Bats are completely blind.", "hallucinated"),
    ("The Sahara is the largest desert in the world.", "hallucinated"),
    ("Shaving makes hair grow back thicker.", "hallucinated"),
    ("Napoleon was extremely short.", "hallucinated"),
    ("Lightning never strikes the same place twice.", "hallucinated"),
    ("Dropping a penny from a skyscraper can kill someone.", "hallucinated"),
    ("The full moon causes increased crime rates.", "hallucinated"),
    
    # Uncertain - prompts that should trigger "I don't know"
    ("What will the stock market do tomorrow?", "uncertain"),
    ("Who will win the next World Cup?", "uncertain"),
    ("What is the exact population of Earth right now?", "uncertain"),
    ("Will AI become sentient by 2030?", "uncertain"),
    ("What is the cure for aging?", "uncertain"),
    ("Are there aliens in the Andromeda galaxy?", "uncertain"),
    ("What will technology look like in 100 years?", "uncertain"),
    ("Is there life after death?", "uncertain"),
    ("What caused the dinosaurs to actually go extinct?", "uncertain"),
    ("Will humans colonize Mars by 2050?", "uncertain"),
    
    # Contradictory - prompts that contain internal contradictions
    ("A square circle has equal sides and no corners.", "contradictory"),
    ("The transparent opaque wall lets light through.", "contradictory"),
    ("A married bachelor lives with his wife.", "contradictory"),
    ("The boiling ice was very hot and cold.", "contradictory"),
    ("A silent scream echoed through the room.", "contradictory"),
    ("The invisible ghost was clearly seen by everyone.", "contradictory"),
    ("The dry water soaked everything.", "contradictory"),
    ("A burning fire that produces no heat.", "contradictory"),
    ("The dead living organism grew rapidly.", "contradictory"),
    ("A stationary moving car drove nowhere.", "contradictory"),
]

if __name__ == "__main__":
    output_path = "D:/ACC LLM Enhancement/data/acc_training/synthetic_conflict_data.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for prompt, label in TRAIN_DATA:
            f.write(json.dumps({"prompt": prompt, "label": label}) + '\n')
    
    print(f"Generated {len(TRAIN_DATA)} synthetic training examples")
    print(f"Saved to: {output_path}")
    
    # Print distribution
    from collections import Counter
    dist = Counter([label for _, label in TRAIN_DATA])
    print("\nLabel distribution:")
    for label, count in dist.items():
        print(f"  {label}: {count}")
