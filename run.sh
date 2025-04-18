#! /bin/bash
cd build
cmake ..
make
cd ..
./build/tsmm 
# pprof --text ./build/tsmm output.prof
# # 或者生成可视化报告
# pprof --web ./build/tsmm output.prof