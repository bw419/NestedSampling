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
	double logl2;
	double stepsize;
	double acceptrate;
	double acceptrate_deriv;

	double* data_real();
	cmplx* data_cmplx();

	sample_data(double* data, double logl, double logl2, double logv, double stepsize, double acceptrate, double acceptrate_deriv) :
		logl(logl), logl2(logl2), logv(logv), weight(exp(logl + logv)),
		stepsize(stepsize), acceptrate(acceptrate),
		acceptrate_deriv(acceptrate_deriv) {
		for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
			data_[i] = data[i];
		}
	}
};


#endif // !NESTED_SAMPLING_H
