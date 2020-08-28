#include "NestedSampling.h"

using namespace std;


cmplx_vec sample_data::data_cmplx() {
    cmplx_vec data_out{};
    for (int i = 0; i < N_FREE_X_CMPTS; ++i) {
        data_out[i] = cmplx(data[i], data[N_FREE_X_CMPTS + i]);
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
        total += pow(uniform_01(rand_gen), 1. / N_CONCURRENT_SAMPLES);
        weight_set.push_back(X_prev - X_i);
    }

    long double w_remaining = 0;//X_i / N_CONCURRENT_SAMPLES;
    for (int i = n_samples - N_CONCURRENT_SAMPLES; i < n_samples; ++i) {
        weight_set.push_back(w_remaining);
    }
    //weight_set.insert(weight_set.end(), N_CONCURRENT_SAMPLES, w_remaining);

    return weight_set;
}


bool dist_cmp(const pair<double, int>& p1, const pair<double, int>& p2) {
    return p1.first > p2.first;
}

void write_outfile_header(ofstream &outfile, const cmplx_vec_prepended &actual_x, const t_mat &transform_mat,
    const string &termination_reason, const double &sampling_time, const double &neighbour_computing_time) {
    outfile << "actual_x_cmpts=";
    for (int j = 1; j < N_FREE_X_CMPTS + 1; ++j) {
        outfile << actual_x[j].real() << ",";
    }
    for (int j = 1; j < N_FREE_X_CMPTS + 1; ++j) {
        outfile << actual_x[j].imag();
        if (j != N_FREE_X_CMPTS) {
            outfile << ",";
        }
    }

    outfile << ";matrix_cmpts_real=";
    for (int j = 0; j < N_IMAGE_CMPTS; ++j) {
        for (int k = 0; k < N_FREE_X_CMPTS + 1; ++k) {
            outfile << transform_mat[j][k].real();
            if (!(j == N_IMAGE_CMPTS - 1) && (k == N_FREE_X_CMPTS)) {
                outfile << ",";
            }
        }
    }
    outfile << ";matrix_cmpts_imag=";
    for (int j = 0; j < N_IMAGE_CMPTS; ++j) {
        for (int k = 0; k < N_FREE_X_CMPTS + 1; ++k) {
            outfile << transform_mat[j][k].imag();
            if (!(j == N_IMAGE_CMPTS - 1) && (k == N_FREE_X_CMPTS)) {
                outfile << ",";
            }
        }
    }

    outfile << "n_image_cmpts=" << N_IMAGE_CMPTS << ";" << endl;
    outfile << "sampling_time=" << sampling_time << ";" << endl;
    outfile << "neighbour_computing_time=" << neighbour_computing_time << ";" << endl;
}

void write_outfile_body(ofstream& outfile, const vector<sample_data> &out_samples, 
    const vector<array<int, N_NEIGHBOURS>> &min_dists) {

    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        outfile << "cmpt_" << i << ",";
    }
    if (COMPUTE_NEIGHBOURS) {
        for (int i = 0; i < N_NEIGHBOURS; ++i) {
            outfile << "adj_" << i << ",";
        }
    }

    outfile << "logl,logv,weight,stepsize,acceptrate,acceptrate_deriv" << endl;

    outfile << scientific << setprecision(10);

    for (int i = 0; i < out_samples.size(); ++i) {
        for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
            outfile << out_samples[i].data[j] << ",";
        }
        if (COMPUTE_NEIGHBOURS) {
            for (int j = 0; j < N_NEIGHBOURS; ++j) {
                outfile << min_dists[i][j] << ",";
            }
        }
        outfile << out_samples[i].logl << "," << out_samples[i].logv << "," << out_samples[i].weight
            << "," << out_samples[i].stepsize << "," << out_samples[i].acceptrate
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
    vector<array<int, N_NEIGHBOURS>>& min_dists_container_to_write) {
    if (LOG_PROGRESS) {
        cout << "finding neighbours... (" << out_samples.size() << " samples)" << endl;
    }

    // connectivity bit.
    for (int i = 0; i < out_samples.size(); ++i) {
        array<pair<double, int>, N_NEIGHBOURS> curr_min_dists{};

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
array<double, N_CONCURRENT_SAMPLES> curr_sample_loglikes{};
vector<sample_data> out_samples;
vector<array<int, N_NEIGHBOURS>> min_dists{};


int main() {

    if (LOG_PROGRESS) {
        cout << "Files: start " << FILE_N_START << ", end " << FILE_N_STOP << endl;
    }

    // loop entire program, ii=filename number
    for (int file_number = FILE_N_START; file_number < FILE_N_STOP; ++file_number) {

        intitialise_phase_reconstruction();

        clock_t start_t = clock();
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
        BallWalkMCMC mcmc = BallWalkMCMC(.1, loglike_from_sample_vec, max(10, N_CONCURRENT_SAMPLES), .5, 0.01, 10);
        //GalileanMCMC mcmc = GalileanMCMC(.1, loglike_from_sample_vec, grad_loglike_from_sample_vec, 
        //                                 max(10, N_CONCURRENT_SAMPLES), .8, 0.1, 100, 0.2);

        int it = 1;
        for (it = 1; it <= N_ITERATIONS; ++it) {

            //for (int j = 0; j < N_CONCURRENT_SAMPLES; j+=1){//N_CONCURRENT_SAMPLES/10) {
            //    cout << curr_sample_loglikes[j] << " | ";
            //}
            //cout << endl;
            //cout << "rate, " << mcmc.acceptance_rate() << " | step size, " << mcmc.step_size() << endl;
            //cout << "---------------------------------------------" << endl;

            if (!(it % 200) && LOG_PROGRESS) {
                cout << "iteration " << it << ", success rate: " << mcmc.acceptance_rate()
                    << ", step size: " << mcmc.step_size() << ", logz: " << log(Z) << endl;
            }

            auto min_L_it = min_element(curr_sample_loglikes.begin(), curr_sample_loglikes.end());
            int min_L_idx = (min_L_it - curr_sample_loglikes.begin());

            //cout << "min idx: " << min_L_idx << endl;
               
            X_curr_est = exp(-(long double)it / N_CONCURRENT_SAMPLES);
            w_est = X_prev_est - X_curr_est;
            Z += exp(*min_L_it) * w_est;

            out_samples.push_back(
                sample_data(
                    current_samples[min_L_idx],
                    *min_L_it, log(w_est),
                    mcmc.step_size(), mcmc.acceptance_rate(),
                    mcmc.acceptance_rate_deriv()
                )
            );

            int start_pt_idx = uniform_rand_sample(rand_gen);

            //cout << "before... " << start_pt_idx << " , " << current_samples[start_pt_idx][0] << ", " << *min_L_it << endl;
            //cout << "start idx: " << start_pt_idx << ", min L idx: " << min_L_idx << endl;
            mcmc.evolve(current_samples, start_pt_idx, min_L_idx, *min_L_it);


            X_prev_est = exp(-(long double)it / N_CONCURRENT_SAMPLES);

            //cout << "after..." << current_samples[min_L_idx][0] << ", " << *min_L_it << endl;
            //cout << "-------------------------------" << endl;


            //cout << "~~~" << endl;
            //mcmc.acceptance_rate();
            //cout << "~~~" << endl;

            if (exp(*min_L_it) * X_curr_est < TERMINATION_PERCENTAGE * Z) {
                termination_reason = "evidence accumulation percentage";
                break;
            }
            if (mcmc.step_size() < TERMINATION_STEPSIZE) {
                termination_reason = "step size";
                break;
            }
            if (it == N_ITERATIONS - 1) {
                termination_reason = "max iterations";
                break;
            }
        }

        if (LOG_PROGRESS) {
            cout << "Terminated (" << termination_reason << "), Z: " << Z << endl;
        }


        if (USE_REMAINING_SAMPLES) {
            if (LOG_PROGRESS) {
                cout << "sorting remaining points..." << endl;
            }

            w_est = (exp(-(long double)(it) / N_CONCURRENT_SAMPLES)) / N_CONCURRENT_SAMPLES;

            array<pair<double, sample_vec>, N_CONCURRENT_SAMPLES> remaining{};
            for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
                remaining[i].first = curr_sample_loglikes[i];
                remaining[i].second = current_samples[i];
            }
            sort(remaining.begin(), remaining.end());

            if (LOG_PROGRESS) {
                cout << "----------------------------------------" << endl;
            }

            for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
                out_samples.push_back(
                    sample_data(
                        remaining[i].second,
                        remaining[i].first, log(w_est),
                        mcmc.step_size(), mcmc.acceptance_rate(),
                        mcmc.acceptance_rate_deriv()
                    )
                );
                Z += exp(curr_sample_loglikes[i]) * w_est;
            }
        }


        sampling_time = (double)(clock() - start_t) / CLOCKS_PER_SEC;
        long double drawn_logZ_std_dev = compute_logZ_uncertainty(out_samples);

        if (LOG_PROGRESS) {
            cout << "iterations: " << it << endl;
            cout << "sampling time : " << sampling_time << "s)" << endl;
            cout << "Z: " << Z << endl;
            cout << "logz: " << log(Z) << " +- " << drawn_logZ_std_dev << endl;
        }


        neighbour_computing_time = 0.;
        if (COMPUTE_NEIGHBOURS) {
            start_t = clock();
            min_dists.resize(out_samples.size());
            compute_neighbours(out_samples, min_dists);
            neighbour_computing_time = (double)(clock() - start_t) / CLOCKS_PER_SEC;
            if (LOG_PROGRESS) {
                cout << "elapsed neighbour computing time: " << neighbour_computing_time << "s" << endl;
            }
        }


        if (LOG_PROGRESS) {
            cout << "writing to file..." << endl;
        }

        ofstream outfile;
        outfile.open(OUT_PATH + to_string(file_number) + ".txt");

        write_outfile_header(outfile, actual_x, transform_mat, termination_reason, sampling_time, neighbour_computing_time);
        write_outfile_body(outfile, out_samples, min_dists);

        outfile.close();


        out_samples.clear();
        min_dists.clear();

        if (LOG_PROGRESS) {
            cout << "----------------------------------------" << endl;
        }
    }
}