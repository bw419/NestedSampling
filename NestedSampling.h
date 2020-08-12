#pragma once

#ifndef NESTED_SAMPLING_H
#define NESTED_SAMPLING_H

#include "Globals.h"
#include "CircBuffer.h"

vector<double> draw_weight_set(size_t n_samples);

struct sample_data {

	double data_[N_SAMPLE_CMPTS] {};
	double weight;
	double logv;
	double logl;
	double stepsize;
	double acceptrate;
	double acceptrate_deriv;

	double* data_real();
	void data_cmplx(cmplx* data_out);

	sample_data(double* data, double logl, double logv, double stepsize, double acceptrate, double acceptrate_deriv) :
		logl(logl), logv(logv), weight(exp(logl + logv)),
		stepsize(stepsize), acceptrate(acceptrate),
		acceptrate_deriv(acceptrate_deriv) {
		for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
			data_[i] = data[i];
		}
	}
};


#endif // !NESTED_SAMPLING_H
