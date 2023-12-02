cd ChampSim

bin/champsim_tracegen --warmup-instructions 200000000 --simulation-instructions 500000000 ../traces/403.gcc-48B.champsimtrace.xz
mv ./llc_trace.txt ../data/llc_trace_403.txt

bin/champsim_tracegen --warmup-instructions 200000000 --simulation-instructions 500000000 ../traces/429.mcf-217B.champsimtrace.xz
mv ./llc_trace.txt ../data/llc_trace_429.txt

bin/champsim_tracegen --warmup-instructions 200000000 --simulation-instructions 500000000 ../traces/437.leslie3d-273B.champsimtrace.xz
mv ./llc_trace.txt ../data/llc_trace_437.txt

bin/champsim_tracegen --warmup-instructions 200000000 --simulation-instructions 500000000 ../traces/471.omnetpp-188B.champsimtrace.xz
mv ./llc_trace.txt ../data/llc_trace_471.txt

bin/champsim_tracegen --warmup-instructions 200000000 --simulation-instructions 500000000 ../traces/483.xalancbmk-736B.champsimtrace.xz
mv ./llc_trace.txt ../data/llc_trace_483.txt
