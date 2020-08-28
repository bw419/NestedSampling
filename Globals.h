#ifndef GLOBALS_H
#define GLOBALS_H

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <math.h>
#include <ctime>
#include <tuple>
#include <array>


#define OUT_PATH "out/samples"
#define LOG_PROGRESS false

#define FILE_N_START 0
#define FILE_N_STOP 1
#define USE_REMAINING_SAMPLES true


#define N_CONCURRENT_SAMPLES 100
#define N_FREE_X_CMPTS 9
#define N_SAMPLE_CMPTS 2*N_FREE_X_CMPTS
#define N_IMAGE_CMPTS 32
#define N_STEPS_PER_SAMPLE 50
#define N_ITERATIONS 100000
#define TERMINATION_PERCENTAGE 1e-06
#define TERMINATION_STEPSIZE 1e-06
#define N_ALTERNATIVE_WEIGHT_SAMPLES 20

#define COMPUTE_NEIGHBOURS true
#define N_NEIGHBOURS 10


using namespace std;
typedef complex<double> cmplx;
typedef array<double, N_SAMPLE_CMPTS> sample_vec;
typedef array<double, N_IMAGE_CMPTS> image_vec;
typedef array<sample_vec, N_CONCURRENT_SAMPLES> sample_collection;
typedef array<cmplx, N_FREE_X_CMPTS> cmplx_vec;
typedef array<cmplx, N_FREE_X_CMPTS + 1> cmplx_vec_prepended;



#define SQRT_2_PI 2.506628275
#define SQRT_2 1.414213562
#define SQRT_3 1.732050808
#define INV_SQRT_2 0.707106781
#define INV_SQRT_3 0.577350269



void overwrite_sample(sample_vec& old_sample, const sample_vec& new_sample);
void print_vec(string name, double* vec, int length);
void normalise_vec(sample_vec& vec);
double get_vec_norm(const sample_vec& vec);

#endif