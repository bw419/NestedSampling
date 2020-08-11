#include "Globals.h"

using namespace std;



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

// assume length > 1
void normalise_vec(double* vec, int length){
    double norm = get_vec_norm(vec, length);
    for (int i = 0; i < length; ++i) {
        vec[i] /= norm;
    }
}

double get_vec_norm(double* vec, int length) {
    double tot = 0;
    for (int i = 0; i < length; ++i) {
        tot += vec[i] * vec[i];
    }
    return sqrt(tot);
}