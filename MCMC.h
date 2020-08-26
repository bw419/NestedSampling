#ifndef MCMC_H
#define MCMC_H

#include "Globals.h"
#include "CircBuffer.h"


class MCMCWalker {

protected:

	MCMCWalker(double initial_step_size_guess, double (*loglike_fn_ptr) (const sample_vec&), size_t buffer_size, double target_acceptance_rate, double k_prop, double k_deriv) :
		ref_step_size_(initial_step_size_guess),
		loglike_fn_(loglike_fn_ptr),
		n_success_buffer_(buffer_size),
		target_acceptance_rate_(target_acceptance_rate),
		acceptance_rate_(target_acceptance_rate),
		acceptance_rate_deriv_(target_acceptance_rate),
		k_prop_(k_prop),
		k_deriv_(k_deriv)
	{
		if (buffer_size < 5 && N_SAMPLE_CMPTS >= 5) {
			throw std::logic_error("Too small buffer");
		}
	}

	double (*loglike_fn_)  (const sample_vec&);

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

	virtual bool step(sample_collection &samples, int idx_to_write, sample_vec &new_pt, double old_loglike, double& new_loglike) = 0;
	void adjust_step_size();

public:
	double calculate_acceptance_rate(bool print_stats);
	double acceptance_rate();
	double acceptance_rate_deriv();

	double step_size() {
		return ref_step_size_ * exp(step_constant_);
	}

	virtual void evolve(sample_collection &samples, int idx_to_evolve, int idx_to_write, double& min_loglike) = 0;
};


class BallWalkMCMC: public MCMCWalker {

	bool step(sample_collection &samples, int idx_to_write, sample_vec &new_pt, double old_loglike, double& new_loglike);

public:
	BallWalkMCMC(double initial_step_size_guess, double (*loglike_fn_ptr) (const sample_vec&), size_t buffer_size, double target_acceptance_rate, double k_prop, double k_deriv) :
		MCMCWalker(initial_step_size_guess, loglike_fn_ptr, buffer_size, target_acceptance_rate, k_prop, k_deriv) {}

	void evolve(sample_collection &samples, int idx_to_evolve, int idx_to_write, double& min_loglike);
};


class GalileanMCMC : public MCMCWalker {

	sample_vec (*grad_loglike_fn_) (const sample_vec&);

	sample_vec velocity_;
	double perturbation_theta_; // in radians
	void perturb_velocity();

	bool step(sample_collection &samples, int idx_to_write, sample_vec &new_pt, double old_loglike, double& new_loglike);

public:
	GalileanMCMC(double initial_step_size_guess, double (*loglike_fn_ptr) (const sample_vec&), sample_vec (*grad_loglike_fn_ptr) (const sample_vec&),
		size_t buffer_size, double target_acceptance_rate, double k_prop, double k_deriv, double perturbation_theta) :
		MCMCWalker(initial_step_size_guess, loglike_fn_ptr, buffer_size, target_acceptance_rate, k_prop, k_deriv), 
		velocity_(), grad_loglike_fn_(grad_loglike_fn_ptr), perturbation_theta_(perturbation_theta) {
	}

	void evolve(sample_collection &samples, int idx_to_evolve, int idx_to_write, double& min_loglike);
};

#endif