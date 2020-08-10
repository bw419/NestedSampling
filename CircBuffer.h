#pragma once

#include "Globals.h"

template <typename T>

class CircularBuffer
{
    static const size_t EMPTY = ~0u;

    size_t next = 0;               // index at which next insert will occur
    size_t head = EMPTY;           // index of next pop, or EMPTY marker
    size_t data_vec_size = 0;
    bool full = false;
public:
    std::vector<T> data;

    CircularBuffer(size_t size)
        : data(size), data_vec_size(size)
    {}

    bool empty() {
        return head == EMPTY;
    }

    size_t size() {
        if (full)
            return data_vec_size;
        if (empty())
            return 0;
        if (next > head)
            return next - head;
        full = true;
        return data_vec_size;
    }

    void push(T val) {
        data[next] = val;
        if (empty())
            head = next;
        if (++next == data.size())
            next = 0;
    }
};