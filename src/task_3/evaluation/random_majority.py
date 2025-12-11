import csv
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

input_csv = 'modernbert-cpt/holistic-formalism-modernbert-cpt-seed-42.csv'

random_preds = []
zero_preds = []

with open(input_csv, newline='') as f:
    reader = csv.DictReader(f)
    if "Gold Labels" not in reader.fieldnames:
        raise ValueError("CSV must have a 'Gold Labels' column")

    for row in reader:
        gold = row["Gold Labels"].strip()
        if len(gold) != 1 or not set(gold).issubset({'0', '1'}):
            raise ValueError(f"Invalid label: {gold}")

        rand_pred = ''.join(random.choice('01') for _ in range(1))

        random_preds.append({'Gold Labels': gold, 'Predicted Labels': rand_pred})
        zero_preds.append({'Gold Labels': gold, 'Predicted Labels': '0'})

with open('random_predictions.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Gold Labels', 'Predicted Labels'])
    writer.writeheader()
    writer.writerows(random_preds)

with open('zero_predictions.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Gold Labels', 'Predicted Labels'])
    writer.writeheader()
    writer.writerows(zero_preds)

print("random_predictions.csv and zero_predictions.csv with seed =", RANDOM_SEED)
