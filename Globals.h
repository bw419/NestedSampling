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


#define OUT_PATH "out/samples_94_"
#define LOG_PROGRESS true
#define LOG_PROGRESS_VERBOSE true

#define FILE_N_START 5
#define FILE_N_STOP 6
#define USE_REMAINING_SAMPLES true

#define PRIOR_RANGE_MAX 3

#define N_CONCURRENT_SAMPLES 300
#define N_FREE_X_CMPTS 9
#define N_SAMPLE_CMPTS 2*N_FREE_X_CMPTS
#define N_IMAGE_CMPTS 300
#define N_STEPS_PER_SAMPLE 50
// Hard cap on number of iterations
#define N_ITERATIONS 1000000
#define TERMINATION_PERCENTAGE 1e-10
// This is the numerical stability limit (x + ~1e-17 = x when x~1)
#define TERMINATION_STEPSIZE 1e-15
#define TERMINATION_SCORE .1
#define N_ALTERNATIVE_WEIGHT_SAMPLES 20


#define COMPUTE_NEIGHBOURS false
#define N_NEIGHBOURS 10

// switch between Ball Walk and Galilean
#define POLYMORPHIC_MCMC false
#define POLYMORPHIC_TRANSITION_FILE_N 5

// for numerical reasons
#define ADJUST_LIKELIHOOD true


using namespace std;
typedef complex<long double> cmplx;
typedef array<long double, N_SAMPLE_CMPTS> sample_vec;
typedef array<long double, N_IMAGE_CMPTS> image_vec;
typedef vector<sample_vec> sample_collection;
typedef array<cmplx, N_FREE_X_CMPTS> cmplx_vec;
typedef array<cmplx, N_FREE_X_CMPTS + 1> cmplx_vec_prepended;



#define SQRT_2_PI 2.506628275
#define SQRT_2 1.414213562
#define SQRT_3 1.732050808
#define INV_SQRT_2 0.707106781
#define INV_SQRT_3 0.577350269



void overwrite_sample(sample_vec& old_sample, const sample_vec& new_sample);
void normalise_vec(sample_vec& vec);
double get_vec_norm(const sample_vec& vec);

#endif