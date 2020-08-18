#include "NestedSampling.h"
#include "Distributions.h"
#include "MCMC.h"

using namespace std;

sample_vec sample_data::data_real() {
    return data_;
}

cmplx_vec sample_data::data_cmplx() {
    cmplx_vec data_out{};
    for (int i = 0; i < N_X_CMPTS; ++i) {
        data_out[i] = cmplx(data_[i], data_[N_X_CMPTS + i]);
    }
    return data_out;
}


vector<double> draw_weight_set(size_t n_samples) {
    vector<double> weight_set{};

    double X_i = 1.;
    double X_prev;

    double total = 0;

    for (int i = 0; i < n_samples - N_CONCURRENT_SAMPLES; ++i) {
        X_prev = X_i;
        X_i *= pow(uniform_01(rand_gen), 1./N_CONCURRENT_SAMPLES);
        total += pow(uniform_01(rand_gen), 1. / N_CONCURRENT_SAMPLES);
        weight_set.push_back(X_prev - X_i);
    }

    double w_remaining = 0;//X_i / N_CONCURRENT_SAMPLES;
    for (int i = n_samples - N_CONCURRENT_SAMPLES; i < n_samples; ++i) {
        weight_set.push_back(w_remaining);
    }
    //weight_set.insert(weight_set.end(), N_CONCURRENT_SAMPLES, w_remaining);

    return weight_set;
}



sample_collection current_samples{};
array<double, N_CONCURRENT_SAMPLES> curr_sample_loglikes{};
vector<sample_data> out_samples;


int main() {

    // loop entire program, ii=filename number
    for (int file_number = 0; file_number < 5; ++file_number) {

        intitialise_phase_reconstruction();

        clock_t start_t = clock();

        // generate initial points
        for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
            overwrite_sample(current_samples[i], gen_prior());
            curr_sample_loglikes[i] = loglike_from_sample_vec(current_samples[i]);
        }
        

        double Z = 0;
        double X_prev_est = 1;
        double X_curr_est, w_est;
        BallWalkMCMC mcmc = BallWalkMCMC(.5, loglike_from_sample_vec, max(10, N_CONCURRENT_SAMPLES), .5, 0.001, 10);
        //GalileanMCMC mcmc = GalileanMCMC(.5, loglike_from_sample_vec, grad_loglike_from_sample_vec, 
        //                                 max(10, N_CONCURRENT_SAMPLES), .8, 0.1, 100, 0.2);

        int it = 1;
        for (it = 1; it <= N_ITERATIONS; ++it) {

            //for (int j = 0; j < N_CONCURRENT_SAMPLES; j+=1){//N_CONCURRENT_SAMPLES/10) {
            //    cout << curr_sample_loglikes[j] << " | ";
            //}
            //cout << endl;
            //cout << "rate, " << mcmc.acceptance_rate() << " | step size, " << mcmc.step_size() << endl;
            //cout << "---------------------------------------------" << endl;

            if (!(it % 100)) {
                //cout << endl;
                cout << "\riteration " << it << ", success rate: "  << mcmc.acceptance_rate() << ", step size: " << mcmc.step_size() << ", logz: " << log(Z) << "            ";
                //cout << endl;
            }

            auto min_L_it = min_element(curr_sample_loglikes.begin(), curr_sample_loglikes.end());
            int min_L_idx = (min_L_it - curr_sample_loglikes.begin());

            //cout << "min idx: " << min_L_idx << endl;

            X_curr_est = exp(-(double)it / N_CONCURRENT_SAMPLES);
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
            

            X_prev_est = exp(-(double)it / N_CONCURRENT_SAMPLES);

            //cout << "after..." << current_samples[min_L_idx][0] << ", " << *min_L_it << endl;
            //cout << "-------------------------------" << endl;


            //cout << "~~~" << endl;
            //mcmc.acceptance_rate();
            //cout << "~~~" << endl;

            if (exp(*min_L_it) * X_curr_est < TERMINATION_PERCENTAGE * Z) {
                ++it;
                cout << "\rTerminated (evidence accumulation percentage), Z: " << Z << "                                               ";
                break;
            }
            if (mcmc.step_size() < TERMINATION_STEPSIZE) {
                ++it;
                cout << "\rTerminated (step size), Z: " << Z << "                                                 ";
                break;
            }
            if (it == N_ITERATIONS - 1) {
                ++it;
                cout << "\rTerminated (max iterations), Z: " << Z << "                                               ";
            }
        }


        cout << endl;
        w_est = (exp(-(double)(it) / N_CONCURRENT_SAMPLES)) / N_CONCURRENT_SAMPLES;
        cout << "\rsorting remaining points...";
        sort(curr_sample_loglikes.begin(), curr_sample_loglikes.end());
        cout << "\r- - - - - - - - - - - - - - - - - -" << endl;

        for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
            out_samples.push_back(
                sample_data(
                    current_samples[i],
                    curr_sample_loglikes[i], log(w_est),
                    mcmc.step_size(), mcmc.acceptance_rate(),
                    mcmc.acceptance_rate_deriv()
                )
            );
            Z += exp(curr_sample_loglikes[i]) * w_est;
        }


        cout << "elapsed time: " << (double)(clock() - start_t)/CLOCKS_PER_SEC << "s" << endl;

        double alternative_logZ_vals[N_ALTERNATIVE_WEIGHT_SAMPLES] {};

        double acc_drawn_logZ = 0;
        for (int i = 0; i < N_ALTERNATIVE_WEIGHT_SAMPLES; ++i) {
            double drawn_Z = 0;
            vector<double> drawn_weight_set = draw_weight_set(out_samples.size());


            for (int j = 0; j < out_samples.size(); ++j) {
                drawn_Z += exp(out_samples[j].logl) * drawn_weight_set[j];
            }

            alternative_logZ_vals[i] = log(drawn_Z);
            acc_drawn_logZ += alternative_logZ_vals[i];
        }

        double drawn_logZ_mean = acc_drawn_logZ / N_ALTERNATIVE_WEIGHT_SAMPLES;
        double drawn_logZ_variance = 0;
        for (int i = 0; i < N_ALTERNATIVE_WEIGHT_SAMPLES; ++i) {
            double delta = alternative_logZ_vals[i] - drawn_logZ_mean;
            drawn_logZ_variance += delta * delta;
        }
        drawn_logZ_variance /= (N_ALTERNATIVE_WEIGHT_SAMPLES-1);
        double drawn_logZ_std_dev = sqrt(drawn_logZ_variance);


        cout << "Z: " << Z << endl;
        cout << "logz: " << log(Z) << " +- " << drawn_logZ_std_dev << endl;
        cout << "\rwriting to file...";

        ofstream outfile;
        outfile.open(OUT_PATH + to_string(file_number) + ".txt");

        for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
            outfile << "cmpt_" << i << ",";
        }
        outfile << "logl,logv,weight,stepsize,acceptrate,acceptrate_deriv" << endl;

        outfile << scientific << setprecision(12);

        for (auto sample_data : out_samples) {
            for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
                outfile << sample_data.data_real()[i] << ",";
            }
            outfile << sample_data.logl << "," << sample_data.logv << "," << sample_data.weight
             << "," << sample_data.stepsize << "," << sample_data.acceptrate
             << "," << sample_data.acceptrate_deriv << endl;
        }

        outfile.close();
        out_samples.clear();
        cout << "\r-----------------------------------" << endl;
    }
}
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
