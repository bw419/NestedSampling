#pragma once

#ifndef NESTED_SAMPLING_H
#define NESTED_SAMPLING_H

#include "Globals.h"
#include "CircBuffer.h"

vector<double> draw_weight_set(size_t n_samples);

struct sample_data {

	sample_vec data_{};
	double weight;
	double logv;
	double logl;
	double stepsize;
	double acceptrate;
	double acceptrate_deriv;

	sample_vec data_real();
	cmplx_vec data_cmplx();

	sample_data(sample_vec data, double logl, double logv, double stepsize, double acceptrate, double acceptrate_deriv) :
		logl(logl), logv(logv), weight(exp(logl + logv)),
		stepsize(stepsize), acceptrate(acceptrate),
		acceptrate_deriv(acceptrate_deriv) {
		for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
			data_[i] = data[i];
		}
	}
};


#endif // !NESTED_SAMPLING_H
