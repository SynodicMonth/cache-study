import threading
import subprocess
import os
import re
import json

# List of all executables
executables = [
    "champsim_drrip_ipstride", "champsim_drrip_sppdev", "champsim_lru_nextline",
    "champsim_lru_vaampm", "champsim_ship_no", "champsim_srrip_ipstride",
    "champsim_srrip_sppdev", "champsim_drrip_nextline", "champsim_drrip_vaampm",
    "champsim_lru_no", "champsim_ship_ipstride", "champsim_ship_sppdev",
    "champsim_srrip_nextline", "champsim_srrip_vaampm", "champsim_drrip_no",
    "champsim_lru_ipstride", "champsim_lru_sppdev", "champsim_ship_nextline",
    "champsim_ship_vaampm", "champsim_srrip_no"
]

# Command parameters
warmup_instructions = "20000000"
simulation_instructions = "50000000"
trace_path = "./traces/astar_23B.trace.xz"

# Output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Function to run the command and capture output
def run_command(executable):
    cmd = f"./ChampSim/bin/{executable} --warmup_instructions {warmup_instructions} --simulation_instructions {simulation_instructions} {trace_path}"
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    # Store the output in a file
    output_file_path = os.path.join(output_dir, f"{executable}_output.txt")
    with open(output_file_path, 'w') as file:
        file.write(result.stdout)

# Number of threads
num_threads = 12 # Adjust this number based on your needs

# Create and start threads
threads = []
for exe in executables:
    t = threading.Thread(target=run_command, args=(exe,))
    threads.append(t)
    t.start()

    # Limit the number of active threads
    if len(threading.enumerate()) >= num_threads:
        for t in threads:
            t.join()  # Wait for some threads to complete
        threads = []

# Wait for all threads to complete
for t in threads:
    t.join()

print("All commands executed.")

# postprocess the output
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

# Get all files in the output directory
files = os.listdir(output_dir)

# Get the hit rate for each file
hit_rates = {}
prefetch_hit_rates = {}

for file in files:
    with open(os.path.join(output_dir, file), 'r') as f:
        str = f.read()
        file_name = file[:-11]
        hit_rates[file_name] = get_hitrate(str)
        prefetch_hit_rates[file_name] = get_prefetch_hitrate(str)
        print(f'{file_name} LLC hit rate: {hit_rates[file_name]} Prefetch hit rate: {prefetch_hit_rates[file_name]}')

# save the data to trace_path.json
with open(os.path.join("data", os.path.basename(trace_path) + ".json"), "w") as f:
    json.dump({"hit_rates": hit_rates, "prefetch_hit_rates": prefetch_hit_rates}, f)