open_project matmul.proj -reset

set_top matmul

add_files matmul.cpp
add_files -tb matmul_test.cpp
add_files -tb matmul.gold.dat

open_solution solution -reset

set_part {virtexuplusHBM}

create_clock -period 5

csim_design

csynth_design

cosim_design

exit
