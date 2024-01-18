import threading
import subprocess
import os
import re
import json

# List of all executables
executables = [
    "champsim_par", "champsim_par_base"
]

# Command parameters
warmup_instructions = "10000000"
simulation_instructions = "50000000"

trace_paths = [
    "./traces/400.perlbench-41B.champsimtrace.xz",
    "./traces/401.bzip2-38B.champsimtrace.xz",
    "./traces/403.gcc-48B.champsimtrace.xz",
    "./traces/429.mcf-22B.champsimtrace.xz",
    "./traces/444.namd-23B.champsimtrace.xz",
    "./traces/445.gobmk-30B.champsimtrace.xz",
]

# Output directory
output_dir = "outputs_par_t"
os.makedirs(output_dir, exist_ok=True)

# Function to run the command and capture output
def run_command(executable, trace_path):
    cmd = f"./ChampSim/bin/{executable} --warmup_instructions {warmup_instructions} --simulation_instructions {simulation_instructions} {trace_path}"
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    # Store the output in a file
    output_file_path = os.path.join(output_dir, f"{executable}__{os.path.basename(trace_path)}__output.txt")
    with open(output_file_path, 'w') as file:
        file.write(result.stdout)

# Number of threads
num_threads = 12 # Adjust this number based on your needs

# Create and start threads
threads = []
for trace_path in trace_paths:
    for exe in executables:
        t = threading.Thread(target=run_command, args=(exe, trace_path))
        threads.append(t)
        t.start()

        # Limit the number of active threads
        if len(threading.enumerate()) > num_threads:
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
    hit_rate = useful / issued if requested != 0 else 0
    return hit_rate

def get_ipc(str):
    ipc = re.findall(r'CPU 0 cumulative IPC: (\d+\.\d+) instructions: (\d+) cycles: (\d+)', str)
    instructions = ipc[0][1]
    cycles = ipc[0][2]
    ipc = float(instructions) / float(cycles)
    return ipc

# Get all files in the output directory
files = os.listdir(output_dir)

# Get the hit rate for each file
hit_rates = {k : {} for k in executables}
prefetch_hit_rates = {k : {} for k in executables}
ipcs = {k : {} for k in executables}

for file in files:
    with open(os.path.join(output_dir, file), 'r') as f:
        str = f.read()
        exe, trace, _ = file.split("__")
        hit_rates[exe][trace] = get_hitrate(str)
        prefetch_hit_rates[exe][trace] = get_prefetch_hitrate(str)
        ipcs[exe][trace] = get_ipc(str)
        print(f'{exe} {trace} LLC hit rate: {hit_rates[exe][trace]} Prefetch hit rate: {prefetch_hit_rates[exe][trace]} IPC: {ipcs[exe][trace]}')

# save the data to data.json
with open(os.path.join("data", "data_par_t.json"), "w") as f:
    json.dump({"hit_rates": hit_rates, "prefetch_hit_rates": prefetch_hit_rates, "ipcs": ipcs}, f)