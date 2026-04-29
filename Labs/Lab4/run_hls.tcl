# Create a new Vitis HLS project
open_project -reset hello_hls_project
set_top add
add_files hello_hls.c
add_files -tb hello_hls_test.c

# Create a solution
open_solution -reset "solution1"
set_part {virtexuplusHBM} 

# Run C simulation
csim_design

# Run High-Level Synthesis (HLS)
csynth_design

# Run co-simulation
cosim_design

exit
