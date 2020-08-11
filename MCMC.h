#pragma once
#include "Globals.h"
#include "CircBuffer.h"


class MCMCWalk {

protected:

	MCMCWalk(double initial_step_size_guess, double (*loglike_fn_ptr) (double*), size_t buffer_size, double target_acceptance_rate, double k_prop, double k_deriv) :
		ref_step_size_(initial_step_size_guess),
		loglike_fn_(loglike_fn_ptr),
		n_success_buffer_(buffer_size),
		target_acceptance_rate_(target_acceptance_rate),
		acceptance_rate_(target_acceptance_rate),
		acceptance_rate_deriv_(target_acceptance_rate),
		k_prop_(k_prop),
		k_deriv_(k_deriv)
	{
		if (buffer_size < 5) {
			throw std::logic_error("Too small buffer");
		}
	}

	double (*loglike_fn_) (double*);

	double ref_step_size_;
	double step_constant_ = 0;
	double target_acceptance_rate_;
	double acceptance_rate_;
	double acceptance_rate_deriv_;
	double k_prop_;
	double k_deriv_;

	int curr_step_number_ = 0;
	int curr_walk_successes_ = 0;

	CircularBuffer<unsigned short> n_success_buffer_;

	virtual bool step(double* const old_pt, double* new_pt, double old_loglike, double& new_loglike) = 0;
	void adjust_step_size();

public:
	double calculate_acceptance_rate(bool print_stats);
	double acceptance_rate();
	double acceptance_rate_deriv();

	double step_size() {
		return ref_step_size_ * exp(step_constant_);
	}

	virtual void evolve(double* all_samples, int idx_to_evolve, int idx_to_write, double& min_loglike) = 0;
};


class BallWalkMCMC: public MCMCWalk {

	bool step(double* const old_pt, double* new_pt, double old_loglike, double& new_loglike);

public:
	BallWalkMCMC(double initial_step_size_guess, double (*loglike_fn_ptr) (double*), size_t buffer_size, double target_acceptance_rate, double k_prop, double k_deriv) :
		MCMCWalk(initial_step_size_guess, loglike_fn_ptr, buffer_size, target_acceptance_rate, k_prop, k_deriv) {}

	void evolve(double* all_samples, int idx_to_evolve, int idx_to_write, double& min_loglike);
};

class GalileanMCMC : public MCMCWalk {

	double velocities_[N_SAMPLE_CMPTS * N_CONCURRENT_SAMPLES];
	bool step(double* const old_pt, double* new_pt, double old_loglike, double& new_loglike);

public:
	GalileanMCMC(double initial_step_size_guess, double (*loglike_fn_ptr) (double*), size_t buffer_size, double target_acceptance_rate, double k_prop, double k_deriv) :
		MCMCWalk(initial_step_size_guess, loglike_fn_ptr, buffer_size, target_acceptance_rate, k_prop, k_deriv), velocities_() {

		for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
			double vec_total = 0;
			for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
				velocities_[i * N_SAMPLE_CMPTS + j] = uniform_01(rand_gen);
				vec_total += velocities_[i * N_SAMPLE_CMPTS + j] *velocities_[i * N_SAMPLE_CMPTS + j];
			}
			vec_total = sqrt(vec_total);
			for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
				velocities_[i * N_SAMPLE_CMPTS + j] /= vec_total;
			}
		}

	}

	void evolve(double* all_samples, int idx_to_evolve, int idx_to_write, double& min_loglike);
};

