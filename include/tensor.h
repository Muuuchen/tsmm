#ifndef TENSOR_H
#define TENSOR_H

#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <numeric>

template <typename T>
class Tensor{
    public:
    Tensor(std::vector<int> shape,bool initialized = false, T min = 0, T max = 10) : _shape(shape){
        _size = std::accumulate(shape.begin(),shape.end(),1, std::multiplies<int>());
        _data.resize(_size,0);
        
        if(initialized){
            std::random_device rd;
            std::mt19937 gen(rd());
            if constexpr (std::is_integral<T>()){
                std::uniform_int_distribution<T> dist(min,max);
                std::generate(_data.begin(),_data.end(), [&](){return dist(gen);});
            } else {
              std::uniform_real_distribution<T> dist(min, max);
              std::generate(_data.begin(), _data.end(),
                            [&]() { return dist(gen); });                                        
            }
        }
    }



    T* data_ptr(){
        return _data.data();
    }

    std::vector<int> shape(){
        return _shape;
    }

    T operator[](int index){
        assert(index < _size);
        return _data[index];
    } 
    int shape(int index){
        assert(index < _shape.size());
        return _shape[index];
    }

    inline int size() { return _size; }

    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
        tensor.print_tensor(os, 0, 0);
        return os;
    }
private:
    std::vector<int> _shape;
    std::vector<T> _data;
    int _size;

    void print_tensor(std::ostream &os, int dim, int offset) const {
        if(dim == _shape.size() - 1){
          os << "[";

          for (int i = 0; i < _shape[dim]; i++) {
            os << _data[offset+i] << " ";
          }
          os << "]";
        } else {
          os << "[";
          int stride = std::accumulate(_shape.begin() + dim + 1, _shape.end(), 1, std::multiplies<int>());
        for (int i = 0; i < _shape[dim]; ++i) {
                print_tensor(os, dim + 1, offset + i * stride);
                if (i < _shape[dim] - 1) os << ",\n"; // 添加空格分隔元素
            }
            os << "]";
        }
    }
};


#endif