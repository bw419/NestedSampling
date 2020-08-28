#include "Distributions.h"

using namespace std;


default_random_engine rand_gen(time(0));
normal_distribution<double> std_normal(0, 1);
uniform_real_distribution<double> uniform_01(0, 1);
uniform_int_distribution<int> uniform_rand_sample(0, N_CONCURRENT_SAMPLES - 1);
normal_distribution<double> uniform_circ(0, INV_SQRT_2);

// prepended with an extra 1 + 0j for a single solution.
cmplx_vec_prepended actual_x{ };//, { 1, 0 }, { 0, 1 }, { .4, .7 }, { -.1, -.9 }, { .4, -.5 }, { 1, -1 } };//{ 1, .5, -.1, .4 };
image_vec observed_y = { {} };

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


t_mat transform_mat{};
void intitialise_phase_reconstruction() {
    cout << scientific << setprecision(2);

    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        for (int j = 0; j < N_FREE_X_CMPTS + 1; ++j) {
            if (i < N_FREE_X_CMPTS + 1) {
                if (i == j) {
                    transform_mat[i][i] = 1;
                }
            }
            else {
                transform_mat[i][j] = 1;// gen_circular_gaussian();
            }
        }
    }


    //for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
    //    for (int j = 0; j < N_X_CMPTS + 1; ++j) {
    //        cout << transform_mat[i * (N_X_CMPTS + 1) + j] << " | ";
    //    }
    //    cout << endl;
    //}

    actual_x[0] = cmplx(1, 0);
    for (int i = 1; i < N_FREE_X_CMPTS + 1; ++i) {
        actual_x[i] = cmplx(-.4, .8);// gen_circular_gaussian();
    }

    if (LOG_PROGRESS) {
        cout << "actual x: " << cmplx(1, 0) << " | ";;

        for (int i = 1; i < N_FREE_X_CMPTS + 1; ++i) {
            cout << actual_x[i] << " | ";
        }
        cout << endl;
    }


    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        cmplx sum = 0;
        for (int j = 0; j < N_FREE_X_CMPTS + 1; ++j) {
            sum += actual_x[j] * transform_mat[i][j];
        }
        observed_y[i] = abs(sum);
    }
}


cmplx_vec sample_to_cmplx(const sample_vec &in) {
    cmplx_vec out{};
    for (int i = 0; i < N_FREE_X_CMPTS; ++i) {
        out[i].real(in[i]);
        out[i].imag(in[i + N_FREE_X_CMPTS]);
    }
    return out;
}


sample_vec cmplx_to_sample(const cmplx_vec &in) {
    sample_vec out{};
    for (int i = 0; i < N_FREE_X_CMPTS; ++i) {
        out[i] = in[i].real();
        out[i + N_FREE_X_CMPTS] = in[i].imag();
    }
    return out;
}


// Nested function calls should be optimised away.
double pr_loglike_from_sample(const sample_vec &v_in) {
    return pr_loglike_from_cmplx(sample_to_cmplx(v_in));
}


double pr_loglike_from_cmplx(const cmplx_vec &v_in) {
    double summed = 0;
    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        cmplx transformed_cmpt = cmplx(1.,0.) * transform_mat[i][0];
        for (int j = 1; j < N_FREE_X_CMPTS + 1; ++j) {
            transformed_cmpt += v_in[j-1] * transform_mat[i][j];
        }
        summed += (observed_y[i] - abs(transformed_cmpt)) * (observed_y[i] - abs(transformed_cmpt));
    }
    return -summed;
}


// Nested function calls should be optimised away.
sample_vec grad_pr_loglike_from_sample(const sample_vec &v_in) {
    return grad_pr_loglike_from_cmplx(sample_to_cmplx(v_in));
}


// proportional to gradient is all that is required
sample_vec grad_pr_loglike_from_cmplx(const cmplx_vec &v_in) {
    array<cmplx, N_IMAGE_CMPTS> transformed{};

    for (int i = 0; i < N_IMAGE_CMPTS; ++i) {
        transformed[i] = cmplx(1., 0.) * transform_mat[i][0];
        for (int j = 1; j < N_FREE_X_CMPTS + 1; ++j) {
            transformed[i] += v_in[j - 1] * transform_mat[i][j];
        }
    }

    sample_vec grad{};
    for (int i = 0; i < N_FREE_X_CMPTS; ++i) {
        cmplx cmplx_grad_term = 0;
        for (int k = 0; k < N_IMAGE_CMPTS; ++k) {
            cmplx_grad_term -= (observed_y[k] - abs(transformed[k])) * (conj(transformed[k]) / abs(transformed[k])) * transform_mat[k][i + 1];
        }
        grad[i] = cmplx_grad_term.real();
        grad[i + N_FREE_X_CMPTS] = -cmplx_grad_term.imag();
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
    return 3 * (2 * uniform_01(rand_gen) - 1); //* pow(1./N_SAMPLE_CMPTS, 0.5) ;
}

sample_vec gen_prior() {
    sample_vec sample{};
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        sample[i] = gen_prior_elem();
    }
    return sample;
}

bool is_elem_in_prior_range(const double &elem) {
    return (elem > -3 && elem < 3);
}

bool is_in_prior_range(const sample_vec &sample) {
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        if (!is_elem_in_prior_range(sample[i])) {
            return false;
        }
    }
    return true;
}
