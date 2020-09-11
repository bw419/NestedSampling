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


#define OUT_PATH "out/samples_10_"
#define LOG_PROGRESS true
#define LOG_PROGRESS_VERBOSE true

#define FILE_N_START 0
#define FILE_N_STOP 3
#define USE_REMAINING_SAMPLES true

#define PRIOR_RANGE_MAX 3

#define N_CONCURRENT_SAMPLES 500
#define N_X_CMPTS 4
#define N_IMAGE_CMPTS 24
#define N_STEPS_PER_SAMPLE 50

// Hard cap on number of iterations
#define N_ITERATIONS 1000000
#define TERMINATION_PERCENTAGE 0.0
// This is the numerical stability limit (x + ~1e-17 = x when x~1)
#define TERMINATION_STEPSIZE 1e-13
#define TERMINATION_SCORE 0.1

#define N_ALTERNATIVE_WEIGHT_SAMPLES 20

#define COMPUTE_NEIGHBOURS false
#define N_NEIGHBOURS 10

// causes python parse errors if not log
#define OUTPUT_LOG_WEIGHTS true

// switch between Ball Walk and Galilean
#define POLYMORPHIC_MCMC false
#define POLYMORPHIC_TRANSITION_FILE_N 5

// for numerical reasons
#define ADJUST_LIKELIHOOD true

#define REAL_VERSION true
#if REAL_VERSION
#define N_SAMPLE_CMPTS N_X_CMPTS
#else
#define N_SAMPLE_CMPTS 2*N_X_CMPTS
#endif


using namespace std;
typedef complex<long double> cmplx;
typedef array<long double, N_SAMPLE_CMPTS> sample_vec;
typedef array<long double, N_IMAGE_CMPTS> image_vec;
typedef vector<sample_vec> sample_collection;
typedef array<cmplx, N_X_CMPTS> cmplx_vec;



#define SQRT_2_PI 2.506628275
#define SQRT_2 1.414213562
#define SQRT_3 1.732050808
#define INV_SQRT_2 0.707106781
#define INV_SQRT_3 0.577350269



void overwrite_sample(sample_vec& old_sample, const sample_vec& new_sample);
void normalise_vec(sample_vec& vec);
void print_vec(const sample_vec& vec);
void print_vec(const cmplx_vec& vec);
double get_vec_norm(const sample_vec& vec);

#endif