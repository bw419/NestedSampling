#include "MCMC.h"
#include "Distributions.h"

using namespace std;


void MCMCWalk::adjust_step_size() {
    if (n_success_buffer_.size() > 0) {
        // acceptance rate too high
        // -> rate_error +ve
        // -> decrease step constant
        // -> decrease step rate = initial size * e^(-step constant)
        double rate_error = calculate_acceptance_rate(false) - target_acceptance_rate_;
        step_constant_ += k_prop_ * rate_error + k_deriv_ * acceptance_rate_deriv_;
    }
}


double MCMCWalk::calculate_acceptance_rate(bool print_stats) {
    double total_successes = 0;
    double total_steps = N_STEPS_PER_SAMPLE * ((double)n_success_buffer_.size());
    total_successes += curr_walk_successes_;
    total_steps += curr_step_number_+1;

    if (total_steps < N_STEPS_PER_SAMPLE/10) {
        return target_acceptance_rate_;
    }

    for (int i = 0; i < n_success_buffer_.size(); ++i) {
        total_successes += n_success_buffer_.data[i];
    }

    acceptance_rate_deriv_ = total_successes / total_steps - acceptance_rate_;
    acceptance_rate_ = total_successes / total_steps;

    if (print_stats) {
        cout << "rate: " << acceptance_rate_ << ", ";
        cout << "step size: " << step_size() << endl;
    }

    return acceptance_rate_;
}


double MCMCWalk::acceptance_rate() {
    return acceptance_rate_;
}
double MCMCWalk::acceptance_rate_deriv() {
    return acceptance_rate_deriv_;
}


bool BallWalkMCMC::step(double* const old_pt, double* new_pt, double min_loglike, double& new_loglike) {
    double sigma = step_size();// *pow(N_SAMPLE_CMPTS, -0.5);

    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        new_pt[i] = old_pt[i] + std_normal(rand_gen) * sigma; //2 * uniform_01(rand_gen) - 1;
    }

    new_loglike = this->loglike_fn_(new_pt);

    //cout << "-- " << old_pt[0] << ", " << new_pt[0] << ", " << new_loglike << ", " << sigma << ":";

    if (new_loglike > min_loglike) {
        if (is_in_prior_range(new_pt)) {
            return true;
        }
        return false;
    }
    else {
        return false;
    }
}


void BallWalkMCMC::evolve(double* point, double& loglike, double min_loglike) {

    curr_walk_successes_ = 0;

    for (curr_step_number_ = 0; curr_step_number_ < N_STEPS_PER_SAMPLE; ++curr_step_number_) {
        bool step_success = false;
        double next_pt[N_SAMPLE_CMPTS]{};
        double next_loglike = 0;

        step_success = step(point, next_pt, min_loglike, next_loglike);

        if (step_success) {
            ++curr_walk_successes_;

            // move the new point into old_pt as the origin of next ball walk step
            overwrite_sample(point, next_pt);
            loglike = next_loglike;
        }

    }
    adjust_step_size();

    n_success_buffer_.push(curr_walk_successes_);
}






bool BallWalkMCMC::step(double* const old_pt, double* new_pt, double min_loglike, double& new_loglike) {
    double sigma = step_size(); // *pow(N_SAMPLE_CMPTS, -0.5);

    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        new_pt[i] = old_pt[i] + std_normal(rand_gen) * sigma; //2 * uniform_01(rand_gen) - 1;
    }

    new_loglike = this->loglike_fn_(new_pt);

    //cout << "-- " << old_pt[0] << ", " << new_pt[0] << ", " << new_loglike << ", " << sigma << ":";

    if (new_loglike > min_loglike) {
        if (is_in_prior_range(new_pt)) {
            return true;
        }
        return false;
    }
    else {
        return false;
    }
}


void BallWalkMCMC::evolve(double* point, double& loglike, double min_loglike) {

    curr_walk_successes_ = 0;
    bool overall_success = false;

    //cout << point[0] << ", " << loglike << endl;

    for (curr_step_number_ = 0; curr_step_number_ < N_STEPS_PER_SAMPLE; ++curr_step_number_) {
        bool step_success = false;
        double next_pt[N_SAMPLE_CMPTS]{};
        double next_loglike = 0;

        step_success = step(point, next_pt, min_loglike, next_loglike);

        if (step_success) {
            overall_success = true;
            ++curr_walk_successes_;

            // move the new point into old_pt as the origin of next ball walk step
            overwrite_sample(point, next_pt);
            loglike = next_loglike;
        }

        if (!overall_success) {
            //cout << next_loglike << "/";
        }


        //if (curr_step_number_ % (int)(N_STEPS_PER_SAMPLE/10) == 0) {}
        //cout << step_size() << endl;
    }
    adjust_step_size();


    //if (!overall_success) {
    //    cout << "!";
    //}
    //cout << point[0] << "~:~:~" <<  endl;

    n_success_buffer_.push(curr_walk_successes_);
}