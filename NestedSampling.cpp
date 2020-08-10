#include "NestedSampling.h"
#include "Distributions.h"
#include "MCMC.h"

using namespace std;

double* sample_data::data_real() {
    return data_;
}
cmplx* sample_data::data_cmplx() {
    cmplx x[N_X_CMPTS]{};

    for (int i = 0; i < N_X_CMPTS; ++i) {
        x[i] = cmplx(data_[i], data_[N_X_CMPTS + i]);
    }

    return x;
}



double current_samples[N_SAMPLE_CMPTS * N_CONCURRENT_SAMPLES]{};
double curr_sample_loglikes[N_CONCURRENT_SAMPLES]{};
vector<sample_data> out_samples;

int main() {

    // loop entire program, ii=filename number
    for (int ii = 15; ii < 25; ++ii) {

        // generate initial points
        for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
            overwrite_sample(&current_samples[i * N_SAMPLE_CMPTS], gen_prior());
            curr_sample_loglikes[i] = loglike_from_sample_vec(&current_samples[i * N_SAMPLE_CMPTS]);
        }


        double Z = 0;
        double X_prev_est = 1;
        double X_curr_est, w_est;
        BallWalkMCMC mcmc = BallWalkMCMC(.5, loglike_from_sample_vec, N_CONCURRENT_SAMPLES, .5, 0.01, 100);//0.01, 100);


        for (int i = 1; i <= N_ITERATIONS; ++i) {

            //for (int j = 0; j < N_CONCURRENT_SAMPLES; j+=N_CONCURRENT_SAMPLES/10) {
            //    cout << curr_sample_loglikes[j] << " | ";
            //}
            //cout << endl;
            //cout << "rate, " << mcmc.acceptance_rate() << " | step size, " << mcmc.step_size() << endl;
            //cout << "---------------------------------------------" << endl;

            if (!(i % 100)) {
                //cout << endl;
                //cout << "iteration " << i << ", success rate: "  << mcmc.acceptance_rate() << ", step size: " << mcmc.step_size() << endl;
                //cout << "logz: " << log(Z) << endl;
            }

            auto min_L_it = min_element(curr_sample_loglikes, curr_sample_loglikes + N_CONCURRENT_SAMPLES);
            int min_L_idx = N_SAMPLE_CMPTS * (min_L_it - curr_sample_loglikes);

            X_curr_est = exp(-(double)i / N_CONCURRENT_SAMPLES);
            w_est = X_prev_est - X_curr_est;
            Z += exp(*min_L_it) * w_est;

            out_samples.push_back(
                sample_data(
                    current_samples + min_L_idx,
                    *min_L_it, log(w_est),
                    mcmc.step_size(), mcmc.acceptance_rate(),
                    mcmc.acceptance_rate_deriv()
                )
            );

            int start_pt_idx = uniform_rand_sample(rand_gen);
            //cout << "before... " << start_pt_idx << " , " << (current_samples + start_pt_idx * N_SAMPLE_CMPTS)[0] << ", " << *min_L_it << endl;
            


            double new_pt[N_SAMPLE_CMPTS]{};
            overwrite_sample(new_pt, current_samples + start_pt_idx * N_SAMPLE_CMPTS);
            double new_loglike = curr_sample_loglikes[start_pt_idx];

            mcmc.evolve(new_pt, new_loglike, *min_L_it);

            // overwrite current minimum likelihood and associated point
            overwrite_sample(current_samples + min_L_idx, new_pt);
            (*min_L_it) = new_loglike;

            X_prev_est = exp(-(double)i / N_CONCURRENT_SAMPLES);

            //cout << "-------------------------------" << endl;
            //cout << "after..." << (current_samples + min_L_idx)[0] << ", " << *min_L_it << endl;


            //cout << "~~~" << endl;
            //mcmc.acceptance_rate();
            //cout << "~~~" << endl;

            if (exp(*min_L_it) * X_curr_est < TERMINATION_PERCENTAGE * Z) {

                cout << "Terminated. " << exp(*min_L_it) << ", " << X_curr_est << ", " << Z << endl;
                break;
            }
        }

        cout << endl << "------------------------" << endl;

        w_est = (exp(-(double)N_ITERATIONS / N_CONCURRENT_SAMPLES)) / N_CONCURRENT_SAMPLES;
        for (int i = 0; i < N_CONCURRENT_SAMPLES; ++i) {
            out_samples.push_back(
                sample_data(
                    current_samples + i * N_SAMPLE_CMPTS,
                    curr_sample_loglikes[i], log(w_est),
                    mcmc.step_size(), mcmc.acceptance_rate(),
                    mcmc.acceptance_rate_deriv()
                )
            );
            Z += exp(curr_sample_loglikes[i]) * w_est;
        }

        cout << "Z: " << Z << endl;
        cout << "logz: " << log(Z) << endl;

        //cout << endl << "------------------------" << endl;

        ofstream outfile;
        outfile.open(OUT_PATH + to_string(ii) + ".txt");

        for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
            outfile << "cmpt_" << i << ",";
        }
        outfile << "logl,logv,weight,stepsize,acceptrate,acceptrate_deriv" << endl;

        outfile << scientific << setprecision(16);

        for (auto sample_data : out_samples) {
            //cout << sample_data.data_real()[0] << ", " << sample_data.data_real()[1] << ": "
            //    << sample_data.logl << ", " << sample_data.logv << ", " << sample_data.weight << endl;


            for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
                outfile << sample_data.data_real()[i] << ",";
            }
            outfile << sample_data.logl << "," << sample_data.logv << "," << sample_data.weight
             << "," << sample_data.stepsize << "," << sample_data.acceptrate
             << "," << sample_data.acceptrate_deriv << endl;
        }

        outfile.close();
        out_samples.clear();
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
