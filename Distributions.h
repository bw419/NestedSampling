#pragma once
#include "Globals.h"

double loglike_from_sample_vec(double* p);
double loglike_from_cmplx(cmplx* p);
double* grad_single_gaussian_loglike_from_sample(double* p);
double single_gaussian_loglike_from_sample(double* p);

double gaussian_sum_loglike_from_sample(double* p);

double gen_prior_elem();
double* gen_prior();
bool is_elem_in_prior_range(double elem);
bool is_in_prior_range(double* sample);
