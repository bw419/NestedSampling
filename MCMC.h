#pragma once
#include "Globals.h"
#include "CircBuffer.h"


class BallWalkMCMC {

	double ref_step_size_;
	double step_constant_ = 0;
	double (*loglike_fn_) (double*);
	double target_acceptance_rate_;
	double acceptance_rate_;
	double acceptance_rate_deriv_;
	double k_prop_;
	double k_deriv_;
	int curr_step_number_ = 0;
	int curr_walk_successes_ = 0;

	CircularBuffer<unsigned short> n_success_buffer_;

	bool step(double* const old_pt, double* new_pt, double old_loglike, double& new_loglike);

	void adjust_step_size();

public:
	BallWalkMCMC(double initial_step_size_guess, double (*loglike_fn_ptr) (double*), size_t buffer_size, double target_acceptance_rate, double k_prop, double k_deriv) :
		ref_step_size_(initial_step_size_guess),
		loglike_fn_(loglike_fn_ptr),
		n_success_buffer_(buffer_size),
		target_acceptance_rate_(target_acceptance_rate),
		acceptance_rate_(target_acceptance_rate),
		acceptance_rate_deriv_(target_acceptance_rate),
		k_prop_(k_prop),
		k_deriv_(k_deriv)
	{
		if (buffer_size <= 5) {
			throw std::logic_error("Too small buffer");
		}
	}

	double calculate_acceptance_rate(bool print_stats);
	double acceptance_rate();
	double acceptance_rate_deriv();

	double step_size() {
		return ref_step_size_ * exp(step_constant_);
	}

	void evolve(double* point, double& loglike, double min_loglike);


};
