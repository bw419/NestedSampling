#include "Globals.h"

using namespace std;

default_random_engine rand_gen(time(0));
normal_distribution<double> std_normal(0, 1);
uniform_real_distribution<double> uniform_01(0, 1);
uniform_int_distribution<int> uniform_rand_sample(0, N_CONCURRENT_SAMPLES - 1);



void overwrite_sample(double* old_sample, double* new_sample) {
    for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
        old_sample[j] = new_sample[j];
    }
}

void print_vec(double* p, string name) {
    cout << name << ": ";
    for (int i = 0; i < N_SAMPLE_CMPTS; ++i) {
        if (i != 0)
            cout << ", ";
        cout << p[i];
    }
    cout << endl;
}
