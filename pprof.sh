#! /bin/bash
# CPUPROFILE=prof.out ./build/tsmm 
pprof --text ./build/tsmm prof.out
# # 或者生成可视化报告
# pprof --web ./build/tsmm prof.out