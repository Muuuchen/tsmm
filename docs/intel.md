
vtune


```sh
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/05b14253-a457-4472-bcf7-d98676542072/intel-vtune-2025.1.0.686_offline.sh
sh ./intel-vtune-2025.1.0.686_offline.sh
source /opt/intel/oneapi/vtune/latest/vtune-vars.sh # 这个是在bashrc里面的`
```

```
vtune -collect hotspots -result-dir hotspots -quiet ./build/tsmm
```

## MKL
路径：
ls /opt/intel/oneapi/mkl/latest/include/
## 使用vtune 后面不需要了


## 或许是容器内特权的问题 换 google preftools
### install
wget https://github.com/gperftools/gperftools/releases/tag/gperftools-2.16
cd
make
make install


See docs/cpuprofile.adoc for information about how to use the CPU
profiler and analyze its output.

As a quick-start, do the following after installing this package:

1) Link your executable with -lprofiler
2) Run your executable with the CPUPROFILE environment var set:
     $ CPUPROFILE=/tmp/prof.out <path/to/binary> [binary args]
3) Run pprof to analyze the CPU usage
     $ pprof <path/to/binary> /tmp/prof.out      # -pg-like text output
     $ pprof --gv <path/to/binary> /tmp/prof.out # really cool graphical output

There are other environment variables, besides CPUPROFILE, you can set
to adjust the cpu-profiler behavior; cf "ENVIRONMENT VARIABLES" below.

The CPU profiler is available on all unix-based systems we've tested;
see INSTALL for more details.  It is not currently available on Windows.

NOTE: CPU profiling doesn't work after fork (unless you immediately
      do an exec()-like call afterwards).  Furthermore, if you do
      fork, and the child calls exit(), it may corrupt the profile
      data.  You can use _exit() to work around this.  We hope to have
      a fix for both problems in the next release of perftools
      (hopefully perftools 1.2).




```sh



wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/dc93af13-2b3f-40c3-a41b-2bc05a707a80/intel-onemkl-2025.1.0.803_offline.sh
sudo sh ./intel-onemkl-2025.1.0.803_offline.sh
```



好吧 最后还是重新起了一下docker
