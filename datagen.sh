cd ChampSim

bin/champsim_tracegen --warmup-instructions 20000000 --simulation-instructions 100000000 ../traces/astar_23B.trace.xz
mv ./llc_trace.txt ../data/llc_trace_astar.txt
