open_project block_mm.proj -reset

set_top blockmatmul

add_files block_mm.cpp
add_files -tb blockmatmul_test.cpp

open_solution solution -reset

set_part {virtexuplusHBM}

create_clock -period 5

csim_design

config_compile -pipeline_loops 0

csynth_design

cosim_design

exit
