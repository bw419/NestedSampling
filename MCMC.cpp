#include "MCMC.h"
#include "Distributions.h"

using namespace std;


void MCMCWalker::adjust_step_size() {
    if (n_success_buffer_.size() > 0) {
        // acceptance rate too high
        // -> rate_error +ve
        // -> decrease step constant
        // -> decrease step rate = initial size * e^(-step constant)
        double rate_error = calculate_acceptance_rate(false) - target_acceptance_rate_;
        double adjust_amount = k_prop_ * rate_error + k_deriv_ * acceptance_rate_deriv_;
        double thresh = 1 ;
        if (abs(adjust_amount) > thresh) {
            //cout << "too much adjustment " << adjust_amount << "  " << adjust_amount / abs(adjust_amount);
            step_constant_ += thresh * adjust_amount/abs(adjust_amount);
        }
        else {
            step_constant_ += adjust_amount;
        }
    }
}


double MCMCWalker::calculate_acceptance_rate(bool print_stats) {
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


    if (total_steps < N_STEPS_PER_SAMPLE) {
        acceptance_rate_deriv_ = 0;
    }

    acceptance_rate_ = total_successes / total_steps;

    if (print_stats) {
        cout << "total successes: " << total_successes << ", ";
        cout << "total steps: " << total_steps << ", ";
        cout << "rate: " << acceptance_rate_ << ", ";
        cout << "step size: " << step_size() << endl;
    }

    return acceptance_rate_;
}


double MCMCWalker::acceptance_rate() {
    return acceptance_rate_;
}
double MCMCWalker::acceptance_rate_deriv() {
    return acceptance_rate_deriv_;
}


bool BallWalkMCMC::step(sample_collection &samples, int idx_to_write, sample_vec &new_pt, double min_loglike, double& new_loglike) {
    double sigma = step_size();// *pow(N_SAMPLE_CMPTS, -0.5);

    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        new_pt[i] = samples[idx_to_write][i] + std_normal(rand_gen) * sigma; //2 * uniform_01(rand_gen) - 1;
    }

    new_loglike = this->loglike_fn_(new_pt);

    if (new_loglike > min_loglike && is_in_prior_range(new_pt)) {
        return true;
    }
    else {
        return false;
    }
}


void BallWalkMCMC::evolve(sample_collection &samples, int idx_to_evolve, int idx_to_write, double& min_pt_loglike) {

    overwrite_sample(samples[idx_to_write], samples[idx_to_evolve]);
    double loglike_thresh = min_pt_loglike;
    double next_loglike = 0;

    //cout << "samples size: " << sizeof(samples) << ", " << sizeof(double) << ", " << (samples).size() / sizeof(double) << endl;

    //print_vec("to evolve", samples, samples.size());

    for (curr_step_number_ = 0; curr_step_number_ < N_STEPS_PER_SAMPLE; ++curr_step_number_) {
        bool step_success = false;
        sample_vec next_pt{};
        next_loglike = 0;

        step_success = step(samples, idx_to_write, next_pt, loglike_thresh, next_loglike);

        if (step_success) {
            ++curr_walk_successes_;

            // move the new point into old_pt as the origin of next ball walk step
            overwrite_sample(samples[idx_to_write], next_pt);
            min_pt_loglike = next_loglike;
        }

    }
    n_success_buffer_.push(curr_walk_successes_);
    curr_step_number_ = -1;
    curr_walk_successes_ = 0;

    adjust_step_size();
}

void GalileanMCMC::perturb_velocity() {
    sample_vec new_v{};
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        new_v[i] = std_normal(rand_gen);
    }
    normalise_vec(new_v);
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        velocity_[i] = velocity_[i] * cos(perturbation_theta_) + new_v[i] * sin(perturbation_theta_);
    }
}


bool GalileanMCMC::step(sample_collection &samples, int idx_to_evolve, sample_vec &new_pt, double min_loglike, double& new_loglike) {
    double step = step_size(); // *pow(N_SAMPLE_CMPTS, -0.5);


    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        new_pt[i] = samples[idx_to_evolve][i] + step * velocity_[i];
    }

    new_loglike = this->loglike_fn_(new_pt) ;


    //cout << "new_loglike: " << new_loglike << ", old: " << this->loglike_fn_(samples + idx_to_evolve) << ", step size: " << step << endl;

    if (new_loglike > min_loglike && is_in_prior_range(new_pt)) {
        ++curr_walk_successes_;
        return true;
    }

    //cout << "\nhad to reflect." << endl;

    // try to reflect.
    // v1 = norm here
    sample_vec v1 = this->grad_loglike_fn_(new_pt);
    normalise_vec(v1);

    //cout << "old v: ";
    double dot_prod_doubled = 0;
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        //cout << velocity_[i] << ", ";
        dot_prod_doubled += 2*velocity_[i] * v1[i];
    }
    //cout << endl;

    //cout << "normal: ";
    //for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
    //    cout << v1[i] << ", ";
    //}
    //cout << endl;

    //cout << "new v: ";

    // now v1 = reflection velocity
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        v1[i] = velocity_[i] - dot_prod_doubled * v1[i];
        //cout << v1[i] << ", ";
    }
    //cout << endl;


    // continue to new point
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        new_pt[i] += v1[i]*step;
    }

    // see if the new location is OK
    new_loglike = this->loglike_fn_(new_pt);
    //cout << "new: " << new_pt[0] << ", min: " << loglike_to_rad(min_loglike) << ", in range: " << is_in_prior_range(new_pt) << endl;
    if (new_loglike > min_loglike && is_in_prior_range(new_pt)) {
        //cout << "good new point!" << endl;
        // confirm new velocity
        for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
            velocity_[i] = v1[i];
            //cout << velocity_[i] << ", ";
        }
        //cout << endl;
        return true;
    }
    //cout << "bad new point!" << endl;

    // reflection failed - reverse the velocity and return.
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        velocity_[i] = -velocity_[i];
        //cout << velocity_[i] << ", ";
    }
    //cout << endl;

    return false;
}


void GalileanMCMC::evolve(sample_collection &samples, int idx_to_evolve, int idx_to_write, double& min_pt_loglike) {

    overwrite_sample(samples[idx_to_write], samples[idx_to_evolve]);
    double loglike_thresh = min_pt_loglike;

    // set isotropic initial velocity for the run.
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        velocity_[i] = std_normal(rand_gen);
    }
    normalise_vec(velocity_);

    // step loop
    for (curr_step_number_ = 0; curr_step_number_ < N_STEPS_PER_SAMPLE; ++curr_step_number_) {
        bool step_accepted = false;
        sample_vec next_pt{};
        double next_loglike = 0;

        step_accepted = step(samples, idx_to_write, next_pt, loglike_thresh, next_loglike);

        if (curr_step_number_ % 10 == 0) {
            perturb_velocity();
        }

        if (step_accepted) {
            // move the new point into old_pt as the origin of next ball walk step
            overwrite_sample(samples[idx_to_write], next_pt);
            min_pt_loglike = next_loglike;
        }
    }
    n_success_buffer_.push(curr_walk_successes_);
    curr_walk_successes_ = 0;
    curr_step_number_ = -1;

    adjust_step_size();
}