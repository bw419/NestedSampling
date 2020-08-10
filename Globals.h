#pragma once

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

using namespace std;

#define OUT_PATH "samples_out"

#define N_CONCURRENT_SAMPLES 500//30
#define N_X_CMPTS 1
#define N_SAMPLE_CMPTS 2*N_X_CMPTS
#define N_STEPS_PER_SAMPLE 500
#define N_ITERATIONS 10000
#define TERMINATION_PERCENTAGE 0.0001

#define SQRT_2_PI 2.506628275
#define SQRT_2 1.414213562

using namespace std;
typedef complex<double> cmplx;

extern default_random_engine rand_gen;
extern normal_distribution<double> std_normal;
extern uniform_real_distribution<double> uniform_01;
extern uniform_int_distribution<int> uniform_rand_sample;


void overwrite_sample(double* old_sample, double* new_sample);
void print_vec(double* p, string name);
