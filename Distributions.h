#pragma once
#include "Globals.h"

double loglike_from_sample_vec(double* p);
double loglike_from_cmplx(cmplx* p);
void grad_loglike_from_sample_vec(double const* p, double* grad_out);
double single_gaussian_loglike_from_sample(double* p);
double gaussian_sum_loglike_from_sample(double* p);


double loglike_to_rad(double loglike);


double gen_prior_elem();
double* gen_prior();
bool is_elem_in_prior_range(double elem);
bool is_in_prior_range(double* sample);
