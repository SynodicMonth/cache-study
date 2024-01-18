import re
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def get_hitrate(str):
    total_access = re.findall(r'LLC TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)', str)
    total = int(total_access[0][0])
    hit = int(total_access[0][1])
    miss = int(total_access[0][2])
    hit_rate = hit / total if total != 0 else 0
    return hit_rate

def get_prefetch_hitrate(str):
    "LLC PREFETCH REQUESTED:        741 ISSUED:        741 USEFUL:        146 USELESS:        139"
    total_access = re.findall(r'LLC PREFETCH REQUESTED:\s+(\d+)\s+ISSUED:\s+(\d+)\s+USEFUL:\s+(\d+)\s+USELESS:\s+(\d+)', str)
    requested = int(total_access[0][0])
    issued = int(total_access[0][1])
    useful = int(total_access[0][2])
    useless = int(total_access[0][3])
    hit_rate = useful / requested if requested != 0 else 0
    return hit_rate

path = "outputs"

# Get all files in the output directory
files = os.listdir(path)

# Get the hit rate for each file
hit_rates = {}
prefetch_hit_rates = {}

for file in files:
    with open(os.path.join(path, file), 'r') as f:
        str = f.read()
        file_name = file[:-11]
        hit_rates[file_name] = get_hitrate(str)
        prefetch_hit_rates[file_name] = get_prefetch_hitrate(str)
        print(f'{file_name} LLC hit rate: {hit_rates[file_name]} Prefetch hit rate: {prefetch_hit_rates[file_name]}')

# save the data
with open("data/401.bzip2-38B.champsimtrace.xz.json", "w") as f:
    json.dump({"hit_rates": hit_rates, "prefetch_hit_rates": prefetch_hit_rates}, f)