#include "Distributions.h"

using namespace std;


default_random_engine rand_gen(time(0));
normal_distribution<double> std_normal(0, 1);
uniform_real_distribution<double> uniform_01(0, 1);
uniform_int_distribution<int> uniform_rand_sample(0, N_CONCURRENT_SAMPLES - 1);
normal_distribution<double> uniform_circ(0, INV_SQRT_2);


// prepended with an extra 1 + 0j for a single solution.
#if REAL_VERSION
sample_vec actual_x{}; //, { 1, 0 }, { 0, 1 }, { .4, .7 }, { -.1, -.9 }, { .4, -.5 }, { 1, -1 } };//{ 1, .5, -.1, .4 };
#else
cmplx_vec actual_x{};
cmplx_vec actual_x_normalised{};
#endif
image_vec observed_y = { {} };
double logl_adjustment = 1.;

cmplx gen_circular_gaussian() {
    return { uniform_circ(rand_gen), uniform_circ(rand_gen) };
}

//-------------- LIKELIHOOD RELATED THINGS ----------------//


// Nested function calls should be optimised away.
double loglike_from_sample_vec(const sample_vec& p) {
    return pr_loglike_from_sample(p);
    //return gaussian_sum_loglike_from_sample(p);
}

sample_vec grad_loglike_from_sample_vec(const sample_vec& p) {
    return grad_pr_loglike_from_sample(p);
}

vector<vector<cmplx>> transform_mat{};
void intitialise_phase_reconstruction() {
    cout << scientific << setprecision(2);
    transform_mat.resize(N_IMAGE_CMPTS);

    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        transform_mat[i].resize(N_X_CMPTS);
        for (int j = 0; j < N_X_CMPTS; ++j) {
        #if REAL_VERSION
            transform_mat[i][j] = std_normal(rand_gen);
        #else
            transform_mat[i][j] = gen_circular_gaussian();
        #endif
        }
    }

    //for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
    //    for (int j = 0; j < N_X_CMPTS; ++j) {
    //        cout << transform_mat[i * N_X_CMPTS + j] << " | ";
    //    }
    //    cout << endl;
    //}


#if REAL_VERSION
    for (int i = 0; i <N_X_CMPTS; ++i) {
        double val = std_normal(rand_gen); //cmplx(-.4, .8);
        while (!(is_elem_in_prior_range(val))) {
            val = std_normal(rand_gen); //cmplx(-.4, .8);
        }
        actual_x[i] = val;
    }
#else
    for (int i = 0; i < N_X_CMPTS; ++i) {
        cmplx val = gen_circular_gaussian(); //cmplx(-.4, .8);
        while (!(is_elem_in_prior_range(val.real())
            && is_elem_in_prior_range(val.imag()))) {
            val = gen_circular_gaussian(); //cmplx(-.4, .8);
        }
        actual_x[i] = val;
    }
    cmplx inverse_first_phase = abs(actual_x[0]) / actual_x[0];
    for (int i = 0; i < N_X_CMPTS; ++i) {
        actual_x_normalised[i] = actual_x[i] * inverse_first_phase;
    }
#endif


    if (LOG_PROGRESS_VERBOSE) {
        cout << "actual x: ";
        for (int i = 0; i < N_X_CMPTS; ++i) {
            cout << actual_x[i] << " | ";
        }
        cout << endl;
    }


    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        cmplx sum = 0;
        for (int j = 0; j < N_X_CMPTS; ++j) {
            sum += actual_x[j] * transform_mat[i][j];
        }
        observed_y[i] = abs(sum);
    }
}


cmplx_vec sample_to_cmplx(const sample_vec &in) {
    cmplx_vec out{};
    for (int i = 0; i < N_X_CMPTS; ++i) {
        out[i].real(in[i]);
        out[i].imag(in[i + N_X_CMPTS]);
    }
    return out;
}


sample_vec cmplx_to_sample(const cmplx_vec &in) {
    sample_vec out{};
    for (int i = 0; i < N_X_CMPTS; ++i) {
        out[i] = in[i].real();
        out[i + N_X_CMPTS] = in[i].imag();
    }
    return out;
}


// Nested function calls should be optimised away.
double pr_loglike_from_sample(const sample_vec &v_in) {
#if !REAL_VERSION
    return pr_loglike_from_cmplx(sample_to_cmplx(v_in));
#else
    // for ensuring sensible exp() output range...
    
    //
    (v_in);

    static bool first_call = true;
    double summed = 0;
    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        double transformed_cmpt = 0;
        for (int j = 0; j < N_X_CMPTS; ++j) {
            transformed_cmpt += v_in[j] * transform_mat[i][j].real();
        }
        summed += (observed_y[i] - abs(transformed_cmpt)) * (observed_y[i] - abs(transformed_cmpt));
    }

    //cout << "-------------->" << -summed << endl;

    if (ADJUST_LIKELIHOOD) {
        if (first_call) {
            logl_adjustment = 1 / summed;
            first_call = false;
            //cout << "first call. adjustment = " << adjustment << endl;
        }
        return -summed * logl_adjustment;
    }


    return -summed;
#endif
}

#if !REAL_VERSION
double pr_loglike_from_cmplx(const cmplx_vec &v_in) {
    // for ensuring sensible exp() output range...

    //print_vec(actual_x);
    //print_vec(v_in);

    static bool first_call = true;
    double summed = 0;
    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        cmplx transformed_cmpt = 0;
        for (int j = 0; j < N_X_CMPTS; ++j) {
            transformed_cmpt += v_in[j] * transform_mat[i][j];
        }
        summed += (observed_y[i] - abs(transformed_cmpt)) * (observed_y[i] - abs(transformed_cmpt));
    }
    //cout << "--->" << summed << endl;

    if (ADJUST_LIKELIHOOD) {
        if (first_call) {
            logl_adjustment = 1 / summed;
            first_call = false;
            //cout << "first call. adjustment = " << adjustment << endl;
        }
        return -summed * logl_adjustment;
    }

    return -summed;
}
#endif


// Nested function calls should be optimised away.
sample_vec grad_pr_loglike_from_sample(const sample_vec &v_in) {
#if !REAL_VERSION
    return grad_pr_loglike_from_cmplx(sample_to_cmplx(v_in));
#else
    return {};
#endif
}


// proportional to gradient is all that is required
sample_vec grad_pr_loglike_from_cmplx(const cmplx_vec &v_in) {
    array<cmplx, N_IMAGE_CMPTS> transformed{};

    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        transformed[i] = 0;
        for (int j = 0; j < N_X_CMPTS; ++j) {
            transformed[i] += v_in[j] * transform_mat[i][j];
        }
    }

    sample_vec grad{};
    for (int i = 0; i < N_X_CMPTS; ++i) {
        cmplx cmplx_grad_term = 0;
        for (int k = 0; k < N_IMAGE_CMPTS; ++k) {
            cmplx_grad_term -= (observed_y[k] - abs(transformed[k])) * (conj(transformed[k]) / abs(transformed[k])) * transform_mat[k][i];
        }
        grad[i] = cmplx_grad_term.real();
        grad[i + N_X_CMPTS] = -cmplx_grad_term.imag();
    }

    return grad;
}


double single_gaussian_loglike_from_sample(const sample_vec &p) {
    double acc = -log(2. * 3.14159265) * N_SAMPLE_CMPTS * 0.5;
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        acc -= p[i] * p[i] / (2.); //(p[i] - i) * (p[i] - i);
    }
    return acc;
}


//double loglike_to_rad(double loglike) {
//    return sqrt(-2 * loglike);
//}


double gaussian_sum_loglike_from_sample(const sample_vec &p) {
    double acc = 0;
    double out = 0.;
    double sigma1 = 0.25;
    double sigma2 = 4.00;

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


sample_vec radially_symmetric_grad_loglike(const sample_vec &p) {
    sample_vec grad_out{};
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        grad_out[i] -= p[i];
    }
    return grad_out;
}


//---------------- PRIOR RELATED THINGS -------------------//
double gen_prior_elem() {
    //return std_normal(rand_gen);
    return PRIOR_RANGE_MAX * (2 * uniform_01(rand_gen) - 1); //* pow(1./N_SAMPLE_CMPTS, 0.5) ;
}

sample_vec gen_prior() {
    sample_vec sample{};
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        sample[i] = gen_prior_elem();
    }
    return sample;
}

bool is_elem_in_prior_range(const double &elem) {
    return (elem > -PRIOR_RANGE_MAX && elem < PRIOR_RANGE_MAX);
}

bool is_in_prior_range(const sample_vec &sample) {
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        if (!is_elem_in_prior_range(sample[i])) {
            return false;
        }
    }
    return true;
}
