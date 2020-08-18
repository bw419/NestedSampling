#include "Globals.h"

using namespace std;




void overwrite_sample(sample_vec &old_sample, const sample_vec &new_sample) {
    for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
        old_sample[j] = new_sample[j];
    }
}

void print_vec(string name, double* vec, int length) {
    cout << name << ": ";
    for (int i = 0; i < length; ++i) {
        if (i != 0)
            cout << ", ";
        cout << vec[i];
    }
    cout << endl;
}

// assume length > 1
void normalise_vec(sample_vec& vec){
    double norm = get_vec_norm(vec);
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] /= norm;
    }
}

double get_vec_norm(const sample_vec& vec) {
    double tot = 0;
    for (int i = 0; i < vec.size(); ++i) {
        tot += vec[i] * vec[i];
    }
    return sqrt(tot);
}