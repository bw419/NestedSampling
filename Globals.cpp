#include "Globals.h"

using namespace std;



void overwrite_sample(sample_vec &old_sample, const sample_vec &new_sample) {
    for (int j = 0; j < N_SAMPLE_CMPTS; ++j) {
        old_sample[j] = new_sample[j];
    }
}


// assume length > 1
void normalise_vec(sample_vec& vec){
    double norm = get_vec_norm(vec);
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] /= norm;
    }
}

void print_vec(const sample_vec& vec) {
    for (int i = 0; i < vec.size(); ++i) {
        cout << vec[i];
        if (i != vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << endl;
}


double get_vec_norm(const sample_vec& vec) {
    double tot = 0;
    for (int i = 0; i < vec.size(); ++i) {
        tot += vec[i] * vec[i];
    }
    return sqrt(tot);
}