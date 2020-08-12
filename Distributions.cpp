#include "Distributions.h"

using namespace std;


default_random_engine rand_gen(time(0));
normal_distribution<double> std_normal(0, 1);
uniform_real_distribution<double> uniform_01(0, 1);
uniform_int_distribution<int> uniform_rand_sample(0, N_CONCURRENT_SAMPLES - 1);




//-------------- LIKELIHOOD RELATED THINGS ----------------//

double single_gaussian_loglike_from_sample(double* p) {
    double acc = -log(2. * 3.14159265) * N_SAMPLE_CMPTS * 0.5;
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        acc -= p[i] * p[i] / (2.); //(p[i] - i) * (p[i] - i);
    }
    return acc;
}

//double loglike_to_rad(double loglike) {
//    return sqrt(-2 * loglike);
//}

void grad_single_gaussian_loglike_from_sample(double const* p, double* grad_out) {
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        grad_out[i] -= p[i]; //(p[i] - i) * (p[i] - i);
    }
}

double gaussian_sum_loglike_from_sample(double* p) {
    double acc = 0;
    double out = 0.;
    double sigma1 = 0.1;
    double sigma2 = 10.00;

    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        acc -= p[i] * p[i] / (2. * sigma1 * sigma1);
    }

    out += exp(acc) / pow(sigma1 * SQRT_2_PI, N_SAMPLE_CMPTS);

    acc = 0;
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        acc -= p[i] * p[i] / (2. * sigma2 * sigma2);
    }

    out += exp(acc)/ pow(sigma2 * SQRT_2_PI, N_SAMPLE_CMPTS);


    return log(out/2.);
}


void grad_loglike_from_sample_vec(double const* p, double* grad_out) {
    return grad_single_gaussian_loglike_from_sample(p, grad_out);
}

double loglike_from_sample_vec(double* p) {
    return single_gaussian_loglike_from_sample(p);
}

double loglike_from_cmplx(cmplx* p) {
    return 0.;
}

//---------------- PRIOR RELATED THINGS -------------------//

double gen_prior_elem() {
    return 5 * (2 * uniform_01(rand_gen) - 1); //* pow(1./N_SAMPLE_CMPTS, 0.5) ;
}

double* gen_prior() {
    static double sample[N_SAMPLE_CMPTS];
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        sample[i] = gen_prior_elem();
    }
    return sample;
}

bool is_elem_in_prior_range(double elem) {
    return (elem > -5 && elem < 5);
}

bool is_in_prior_range(double* sample) {
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        if (!is_elem_in_prior_range(sample[i])) {
            return false;
        }
    }
    return true;
}
