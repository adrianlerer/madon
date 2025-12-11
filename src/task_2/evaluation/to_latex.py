import csv
from collections import defaultdict

csv_file = 'filtered_multi.csv'

experiments = defaultdict(list)

with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        experiment = row[0]
        experiments[experiment].append(row)

for experiment, rows in experiments.items():
    print(f"Experiment: {experiment}")
    for i, row in enumerate(rows):
        label = row[1]
        values = row[2:11]
        # each value must be in this format xx.x (only one decimal place), must also be rounded in case of floating point imprecision
        values = [f"{float(value):.1f}" for value in values]
        values_str = " & ".join(values)
        prefix = label if i == 0 else "& " + label
        print(f"{prefix} & {values_str} \\\\")
    print()
