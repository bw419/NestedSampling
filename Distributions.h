#pragma once
#include "Globals.h"

extern default_random_engine rand_gen;
extern normal_distribution<double> std_normal;
extern uniform_real_distribution<double> uniform_01;
extern normal_distribution<double> uniform_circ;
extern uniform_int_distribution<int> uniform_rand_sample;
cmplx gen_circular_gaussian();

extern cmplx_vec_prepended actual_x;
extern image_vec observed_y;


double loglike_from_sample_vec(const sample_vec &p);
sample_vec grad_loglike_from_sample_vec(const sample_vec &p);


cmplx_vec sample_to_cmplx(const sample_vec &in);
sample_vec cmplx_to_sample(const cmplx_vec &in);

void intitialise_phase_reconstruction();
double pr_loglike_from_cmplx(const cmplx_vec &v_in);
double pr_loglike_from_sample(const sample_vec &v_in);


double single_gaussian_loglike_from_sample(const sample_vec &p);
double gaussian_sum_loglike_from_sample(const sample_vec &p);
sample_vec radially_symmetric_grad_loglike(const sample_vec &p);


//double loglike_to_rad(double loglike);

double gen_prior_elem();
sample_vec gen_prior();
bool is_elem_in_prior_range(const double &elem);
bool is_in_prior_range(const sample_vec &sample);
