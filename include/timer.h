#ifndef TIMER_H
#define TIMER_H

#include <cassert>
#include <chrono>
#include <iostream>
#include <ratio>
#include <string>

class Timer {
private:
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  std::string name;
  bool running;
  int M, N, K;

public:
  Timer(const std::string &name = "tsmm", int m = 0, int n = 0, int k = 0)
      : name(name), M(m), N(n), K(k), running(false) {}
  void start() {
    assert(!running);
    start_time = std::chrono::high_resolution_clock::now();
    running = true;
  }
  void stop() {
    assert(running);
    end_time = std::chrono::high_resolution_clock::now();
    running = false;
  }
  double duration_milliseconds() const {
    if (running) {
      auto now = std::chrono::high_resolution_clock::now();
      return std::chrono::duration<double, std::milli>(now - start_time)
          .count();
    }
    return std::chrono::duration<double, std::milli>(end_time - start_time)
        .count();
  }

  inline double duration_seconds() const {
    return duration_milliseconds() / 1000.0;
  }

  //[TODO] FLOPS 的算法
  double flops() const { return (2.0 * M * N * K) / duration_seconds(); }
  // 加上颜色
  void print_result() const {
    std::cout << "\033[32mTimer: " << name << "\033[0m" << std::endl;
    std::cout << "Duration: " << duration_milliseconds() << " ms" << std::endl;
    std::cout << "FLOPS: " << flops() << " GFLOPS" << std::endl;
    std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;
    std::cout << "\033[32m----------------------------------------\033[0m"
              << std::endl;
    fflush(stdout);
  }
  void set_name(std::string str) { name = str; }

  void set_dim(int m, int n, int k) {
    M = m;
    N = n;
    K = k;
  }
};

#endif