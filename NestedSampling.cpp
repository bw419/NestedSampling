#include "NestedSampling.h"

using namespace std;


cmplx_vec sample_data::data_cmplx() {
    cmplx_vec data_out{};
    for (int i = 0; i < N_X_CMPTS; ++i) {
        data_out[i] = cmplx(data[i], data[N_X_CMPTS + i]);
    }
    return data_out;
}


vector<long double> draw_weight_set(size_t n_samples) {
    vector<long double> weight_set{};

    long double X_i = 1.;
    long double X_prev;

    long double total = 0;

    for (int i = 0; i < n_samples - N_CONCURRENT_SAMPLES; ++i) {
        X_prev = X_i;
        X_i *= pow(uniform_01(rand_gen), 1./N_CONCURRENT_SAMPLES);
        total += pow(uniform_01(rand_gen), 1./N_CONCURRENT_SAMPLES);
        weight_set.push_back(X_prev - X_i);
    }

    long double w_remaining = 0;//X_i / N_CONCURRENT_SAMPLES;
    for (int i = n_samples - N_CONCURRENT_SAMPLES; i < n_samples; ++i) {
        weight_set.push_back(w_remaining);
    }
    //weight_set.insert(weight_set.end(), N_CONCURRENT_SAMPLES, w_remaining);

    return weight_set;
}


double get_score(const sample_vec& sample) {
    double acc = 0;
#if REAL_VERSION
    for (int k = 0; k < N_X_CMPTS; ++k) {
        acc += pow(abs(sample[k]) - abs(actual_x[k]), 2);
    }
#else

    auto s = sample_to_cmplx(sample);
    cmplx first_cmpt_inv_phase = abs(s[0]) / s[0];//(long double)1. / s[0];
    for (int k = 0; k < N_X_CMPTS; ++k) {
        acc += pow(abs(s[k]*first_cmpt_inv_phase - actual_x_normalised[k]), 2);
    }
#endif
    return sqrt(acc);
}


double get_score_of_mean(const sample_collection& current_samples) {
    sample_vec mean_sample{};
    array<double, N_CONCURRENT_SAMPLES> multipliers{};

    for (int k = 0; k < N_CONCURRENT_SAMPLES; ++k) {
        sample_vec s{};

#if !REAL_VERSION
        cmplx_vec c_s = sample_to_cmplx(current_samples[k]);

        cmplx first_cmpt_inv_phase = abs(c_s[0]) / c_s[0];
        for (int j = 0; j < N_X_CMPTS; ++j) {
            c_s[j] *= first_cmpt_inv_phase;
        }
        s = cmplx_to_sample(c_s);


#else
        overwrite_sample(s, current_samples[k]);
#endif

        for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
            mean_sample[j] += s[j];
        }
    }

    for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
        mean_sample[j] /= N_CONCURRENT_SAMPLES;
    }
    return get_score(mean_sample);
}



bool dist_cmp(const pair<double, int>& p1, const pair<double, int>& p2) {
    return p1.first > p2.first;
}

#if REAL_VERSION
void write_outfile_header(ofstream &outfile, const sample_vec &actual_x, const vector<vector<cmplx>> &transform_mat,
#else
void write_outfile_header(ofstream& outfile, const cmplx_vec& actual_x, const vector<vector<cmplx>>& transform_mat,
#endif
    const long double &logZ, const
    long double &logZ_std_dev, const string& termination_reason,
    const double &sampling_time, const double &neighbour_computing_time, const int &n_iterations) {

#if REAL_VERSION
    outfile << "actual_x_cmpts=";
    for (int j = 0; j < N_X_CMPTS; ++j) {
        outfile << actual_x[j];
        if (j != N_X_CMPTS - 1) {
            outfile << ",";
        }
    }
    outfile << ";matrix_cmpts_real=";
    for (int j = 0; j < N_IMAGE_CMPTS; ++j) {
        for (int k = 0; k < N_X_CMPTS; ++k) {
            outfile << transform_mat[j][k];
            if (!((j == N_IMAGE_CMPTS - 1) && (k == N_X_CMPTS - 1))) {
                outfile << ",";
        }
    }
}
    outfile << ";matrix_cmpts_imag=";
    for (int j = 0; j < N_IMAGE_CMPTS; ++j) {
        for (int k = 0; k < N_X_CMPTS + 1; ++k) {
            outfile << 0.;
            if (!((j == N_IMAGE_CMPTS - 1) && (k == N_X_CMPTS - 1))) {
                outfile << ",";
            }
        }
    }
    outfile << ";real=1;";
#else
    outfile << "actual_x_cmpts=";
    for (int j = 0; j < N_X_CMPTS; ++j) {
        outfile << actual_x[j].real() << ",";
    }
    for (int j = 0; j < N_X_CMPTS; ++j) {
        outfile << actual_x[j].imag();
        if (j != N_X_CMPTS - 1) {
            outfile << ",";
        }
    }


    outfile << ";matrix_cmpts_real=";
    for (int j = 0; j < N_IMAGE_CMPTS; ++j) {
        for (int k = 0; k < N_X_CMPTS; ++k) {
            outfile << transform_mat[j][k].real();
            if (!((j == N_IMAGE_CMPTS - 1) && (k == N_X_CMPTS - 1))) {
                outfile << ",";
            }
        }
    }
    outfile << ";matrix_cmpts_imag=";
    for (int j = 0; j < N_IMAGE_CMPTS; ++j) {
        for (int k = 0; k < N_X_CMPTS + 1; ++k) {
            outfile << transform_mat[j][k].imag();
            if (!((j == N_IMAGE_CMPTS - 1) && (k == N_X_CMPTS - 1))) {
                outfile << ",";
            }
        }
    }
    outfile << ";real=0;";

#endif

    outfile << "logZ=" << logZ << ";";
    outfile << "logZ_std_dev=" << logZ_std_dev << ";";
    outfile << "n_image_cmpts=" << N_IMAGE_CMPTS << ";";
    outfile << "n_concurrent_samples=" << N_CONCURRENT_SAMPLES << ";";
    outfile << "logl_adjustment_factor=" << logl_adjustment << ";";
    outfile << "sampling_time=" << sampling_time << ";";
    outfile << "neighbour_computing_time=" << neighbour_computing_time << ";";
    outfile << "n_iterations=" << n_iterations << ";";
    outfile << "termination_reason=" << termination_reason << ";";
}

void write_outfile_body(ofstream& outfile, const vector<sample_data> &out_samples, 
    const vector<vector<int>> &min_dists) {

    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        outfile << "cmpt_" << i << ",";
    }
    if (COMPUTE_NEIGHBOURS) {
        for (int i = 0; i < N_NEIGHBOURS; ++i) {
            outfile << "adj_" << i << ",";
        }
    }

    outfile << "logl,logv,";
    if (OUTPUT_LOG_WEIGHTS) {
        outfile << "logw,";
    }
    else {
        outfile << "weight,";
    }
    outfile << "stepsize,acceptrate,acceptrate_deriv" << endl;

    outfile << scientific << setprecision(14); // any more precise than this and numerical stability becomes an issue

    for (int i = 0; i < out_samples.size(); ++i) {
        for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
            outfile << out_samples[i].data[j] << ",";
        }
        if (COMPUTE_NEIGHBOURS) {
            for (int j = 0; j < N_NEIGHBOURS; ++j) {
                outfile << min_dists[i][j] << ",";
            }
        }
        outfile << out_samples[i].logl << "," << out_samples[i].logv << ",";
            
        if (OUTPUT_LOG_WEIGHTS) {
            outfile << log(out_samples[i].weight);
        }
        else {
            outfile << out_samples[i].weight;
        }


        outfile << "," << out_samples[i].stepsize << "," << out_samples[i].acceptrate
            << "," << out_samples[i].acceptrate_deriv << endl;
    }

}


double compute_logZ_uncertainty(const vector<sample_data>& out_samples) {

    long double alternative_logZ_vals[N_ALTERNATIVE_WEIGHT_SAMPLES]{};
    long double acc_drawn_logZ = 0;

    for (int i = 0; i < N_ALTERNATIVE_WEIGHT_SAMPLES; ++i) {
        long double drawn_Z = 0;
        vector<long double> drawn_weight_set = draw_weight_set(out_samples.size());


        for (int j = 0; j < out_samples.size(); ++j) {
            drawn_Z += exp(out_samples[j].logl) * drawn_weight_set[j];
        }

        alternative_logZ_vals[i] = log(drawn_Z);
        acc_drawn_logZ += alternative_logZ_vals[i];
    }

    long double drawn_logZ_mean = acc_drawn_logZ / N_ALTERNATIVE_WEIGHT_SAMPLES;
    long double drawn_logZ_variance = 0;
    for (int i = 0; i < N_ALTERNATIVE_WEIGHT_SAMPLES; ++i) {
        long double delta = alternative_logZ_vals[i] - drawn_logZ_mean;
        drawn_logZ_variance += delta * delta;
    }
    drawn_logZ_variance /= (N_ALTERNATIVE_WEIGHT_SAMPLES - 1);

    return drawn_logZ_variance;
}


void compute_neighbours(const vector<sample_data>& out_samples,
    vector<vector<int>>& min_dists_container_to_write) {
    if (LOG_PROGRESS) {
        cout << "finding neighbours... (" << out_samples.size() << " samples)" << endl;
    }

    // connectivity bit.
    for (int i = 0; i < out_samples.size(); ++i) {
        vector<pair<double, int>> curr_min_dists{};
        curr_min_dists.resize(N_NEIGHBOURS);

        for (int j = 0; j < out_samples.size(); ++j) {
            double curr_dist = 0;

            if (i != j) {
                for (int k = 0; k < N_SAMPLE_CMPTS; ++k) {
                    double cmpt = (out_samples[i].data[k] - out_samples[j].data[k]);
                    curr_dist += cmpt * cmpt;
                }
                curr_dist = sqrt(curr_dist);
            }
            else {
                curr_dist = HUGE_VAL;
            }

            // sorting maintains the largest element at the front
            if (j >= N_NEIGHBOURS) {
                if (j == N_NEIGHBOURS) {
                    sort(curr_min_dists.begin(), curr_min_dists.end(), dist_cmp);
                }
                if (curr_dist < curr_min_dists.front().first) {
                    curr_min_dists.front() = pair<double, int>(curr_dist, j);
                    sort(curr_min_dists.begin(), curr_min_dists.end(), dist_cmp);
                }
            }
            else {
                curr_min_dists[j].first = curr_dist;
                curr_min_dists[j].second = j;
            }
        }
        min_dists_container_to_write[i].resize(N_NEIGHBOURS);
        for (int j = 0; j < N_NEIGHBOURS; ++j) {
            min_dists_container_to_write[i][j] = curr_min_dists[j].second;
        }
    }
}


///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////


sample_collection current_samples{};
vector<double> curr_sample_loglikes{};
vector<sample_data> out_samples{};
vector<vector<int>> min_dists{};


int main() {
    current_samples.resize(N_CONCURRENT_SAMPLES);
    curr_sample_loglikes.resize(N_CONCURRENT_SAMPLES);
    out_samples.reserve(N_ITERATIONS + N_CONCURRENT_SAMPLES);

    if (LOG_PROGRESS_VERBOSE) {
        cout << "Files: start " << FILE_N_START << ", end " << FILE_N_STOP << endl;
    }

    // loop entire program, ii=filename number
    for (int file_number = FILE_N_START; file_number < FILE_N_STOP; ++file_number) {

        intitialise_phase_reconstruction();

        string termination_reason;
        double sampling_time;
        double neighbour_computing_time;


        // generate initial points
        for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
            overwrite_sample(current_samples[i], gen_prior());
            curr_sample_loglikes[i] = loglike_from_sample_vec(current_samples[i]);
        }


        long double Z = 0;
        long double X_prev_est = 1;
        long double X_curr_est, w_est;

        unique_ptr<MCMCWalker> mcmc;
        if (POLYMORPHIC_MCMC && file_number < POLYMORPHIC_TRANSITION_FILE_N) {
            mcmc = unique_ptr<MCMCWalker>(
                new GalileanMCMC(.1, loglike_from_sample_vec, grad_loglike_from_sample_vec,
                    N_CONCURRENT_SAMPLES, .5, 0.1, 100, 0.1));
            cout << "[galilean] ";
        }
        else {
            mcmc = unique_ptr<MCMCWalker>(
                new BallWalkMCMC(.5, loglike_from_sample_vec, N_CONCURRENT_SAMPLES, .5, 0.1, 100));
            cout << "[ball walk] ";
        }
        cout << "File number: " << file_number + 1 << "/" << FILE_N_STOP;
        cout << ", Max iterations: " << N_ITERATIONS << endl;
        clock_t start_t = clock();

        int it;
        for (it = 1; it <= N_ITERATIONS; ++it) {

            //for (int j = 0; j < N_CONCURRENT_SAMPLES; j+=1){//N_CONCURRENT_SAMPLES/10) {
            //    cout << curr_sample_loglikes[j] << " | ";
            //}
            //cout << endl;
            //cout << "rate, " << mcmc->acceptance_rate() << " | step size, " << mcmc->step_size() << endl;
            //cout << "---------------------------------------------" << endl;

            if (!(it % 10000) && LOG_PROGRESS_VERBOSE) {
                cout << "[" << N_X_CMPTS << "->" << N_IMAGE_CMPTS << "] iteration " << it << ", success rate: " << mcmc->acceptance_rate()
                    << ", step size: " << mcmc->step_size() << ", Z: " << Z << ", mean score: " << get_score_of_mean(current_samples) << endl;
            }

            auto min_L_it = min_element(curr_sample_loglikes.begin(), curr_sample_loglikes.end());
            int min_L_idx = (min_L_it - curr_sample_loglikes.begin());

            //cout << "min idx: " << min_L_idx << endl;
               
            X_curr_est = exp(-(long double)it / N_CONCURRENT_SAMPLES);
            w_est = X_prev_est - X_curr_est;
            Z += exp(*min_L_it) * w_est;

            //if (!(it % 100)) {
            //    cout << *min_L_it << ", " << exp(*min_L_it + log(w_est)) << endl;
            //}
            out_samples.push_back(
                sample_data(
                    current_samples[min_L_idx],
                    *min_L_it, log(w_est),
                    mcmc->step_size(), mcmc->acceptance_rate(),
                    mcmc->acceptance_rate_deriv()
                )
            );


            int start_pt_idx = uniform_rand_sample(rand_gen);

            //print_vec(cmplx_to_sample(actual_x));
            //cout << "-------------------------------" << endl;
            //print_vec(current_samples[min_L_idx]);
            //cout << "before... " << start_pt_idx << " , " << current_samples[start_pt_idx][0] << ", " << *min_L_it << endl;
            //cout << "start idx: " << start_pt_idx << ", min L idx: " << min_L_idx << endl;

            mcmc->evolve(current_samples, start_pt_idx, min_L_idx, *min_L_it);

            X_prev_est = exp(-(long double)it / N_CONCURRENT_SAMPLES);
            
            //print_vec(current_samples[min_L_idx]);
            //cout << "after..." << current_samples[min_L_idx][0] << ", " << *min_L_it << endl;
            //cout << "-------------------------------" << endl;

            //cout << "~~~" << endl;
            //cout << mcmc->acceptance_rate() << endl;
            //cout << "~~~" << endl;

            //if (log(Z) > -100.0) {
            //    termination_reason = "likelihood threshold";
            //    break;
            //}
            if (!(it % (N_CONCURRENT_SAMPLES))) {
                if (get_score_of_mean(current_samples) < TERMINATION_SCORE) {
                    //cout << "------------------ scores ------------------" << endl;
                    termination_reason = "mean score below threshold";
                    break;
                }
            }
            if (exp(*min_L_it) * X_curr_est < TERMINATION_PERCENTAGE * Z) {
                //cout << "------------------ evidence ------------------" << endl;
                termination_reason = "evidence accumulation percentage";
                break;
            }
            if (mcmc->step_size() < TERMINATION_STEPSIZE) {
                // For numerical stability reasons.
                //cout << "------------------ step size ------------------" << endl;
                termination_reason = "step size";
                break;
            }
            if (it == N_ITERATIONS - 1) {
                //cout << "------------------ max its ------------------" << endl;
                termination_reason = "max iterations";
                break;
            }
        }


        if (USE_REMAINING_SAMPLES) {
            if (LOG_PROGRESS_VERBOSE) {
                cout << "sorting remaining points..." << endl;
            }

            w_est = (exp(-(long double)(it) / N_CONCURRENT_SAMPLES)) / N_CONCURRENT_SAMPLES;

            vector<pair<long double, sample_vec>> remaining{};
            remaining.resize(N_CONCURRENT_SAMPLES);
            for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
                remaining[i].first = curr_sample_loglikes[i];
                remaining[i].second = current_samples[i];
            }
            sort(remaining.begin(), remaining.end());

            if (LOG_PROGRESS_VERBOSE) {
                cout << "----------------------------------------" << endl;
            }

            for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
                out_samples.push_back(
                    sample_data(
                        remaining[i].second,
                        remaining[i].first, log(w_est),
                        mcmc->step_size(), mcmc->acceptance_rate(),
                        mcmc->acceptance_rate_deriv()
                    )
                );
                Z += exp(curr_sample_loglikes[i]) * w_est;
            }
        }


        sampling_time = (double)(clock() - start_t) / CLOCKS_PER_SEC;
        long double drawn_logZ_std_dev = compute_logZ_uncertainty(out_samples);

        if (LOG_PROGRESS) {
            cout << "Terminated (" << termination_reason << "), logZ: " << log(Z) << ", time: " << sampling_time << "s" << endl;
        }

        if (LOG_PROGRESS_VERBOSE) {
            cout << "iterations: " << it << endl;
            cout << "Z: " << Z << endl;
            cout << "logz: " << log(Z) << " +- " << drawn_logZ_std_dev << endl;
        }


        neighbour_computing_time = 0.;
        if (COMPUTE_NEIGHBOURS) {
            start_t = clock();
            min_dists.resize(out_samples.size());
            compute_neighbours(out_samples, min_dists);
            neighbour_computing_time = (double)(clock() - start_t) / CLOCKS_PER_SEC;
            if (LOG_PROGRESS_VERBOSE) {
                cout << "elapsed neighbour computing time: " << neighbour_computing_time << "s" << endl;
            }
        }


        ofstream outfile;
        string f_name = OUT_PATH + to_string(file_number) + ".txt";
        outfile.open(f_name);

        if (LOG_PROGRESS_VERBOSE) {
            cout << "writing to file \"" << f_name << "\"" << endl;
        }

        write_outfile_header(outfile, actual_x, transform_mat, log(Z), drawn_logZ_std_dev, termination_reason, sampling_time, neighbour_computing_time, it);
        write_outfile_body(outfile, out_samples, min_dists);

        outfile.close();

        out_samples.clear();
        min_dists.clear();


        if (LOG_PROGRESS_VERBOSE) {
            cout << "----------------------------------------" << endl;
        }
    }
}