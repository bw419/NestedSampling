#ifndef NESTED_SAMPLING_H
#define NESTED_SAMPLING_H

#include "Globals.h"
#include "CircBuffer.h"
#include "Distributions.h"
#include "MCMC.h"

vector<long double> draw_weight_set(size_t n_samples);

struct sample_data {

	sample_vec data{};
	long double weight;
	double logv;
	double logl;
	double stepsize;
	double acceptrate;
	double acceptrate_deriv;

	cmplx_vec data_cmplx();

	sample_data(sample_vec data_in, double logl, double logv, double stepsize, double acceptrate, double acceptrate_deriv) :
		logl(logl), logv(logv), weight(exp((long double)(logl + logv))),
		stepsize(stepsize), acceptrate(acceptrate),
		acceptrate_deriv(acceptrate_deriv) {
		for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
			data[i] = data_in[i];
		}
	}
};


#endif // !NESTED_SAMPLING_H
