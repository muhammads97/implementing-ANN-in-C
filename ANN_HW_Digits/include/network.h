#ifndef NETWORK_H
#define NETWORK_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include "InputHandle.h"
#define rando() ((double)rand()/((double)RAND_MAX+1))

class network
{
    public:
        network();
        void init_network(int nInput, int nOutput, int nHidden, int nPattern);
        void set_training(double** input, double** target);
        void set_constants(double eta, double alpha, double smallwt);
        void set_nEpoch(int nEpoch);
        void start_training();
        void store_weights();

        void set_test(double** testInputs, double** testTargets, int nTests);
        void run_test();
        void print_res();
    protected:
    private:
        double** weightIH;
        double** weightHO;
        double** sumH;
        double** input;
        double** target;
        double** hidden;
        double** sumO;
        double** output;
        double** DeltaWeightIH;
        double** DeltaWeightHO;
        double* deltaO;
        double* sumDOW;
        double* deltaH;
        int* ranpat;
        int nInput, nHidden, nOutput, nPattern;
        double Error, eta, alpha, smallwt;
        int i, j, k, p, np, op, epoch;
        int nEpoch;

        double** testInputs;
        double** testOutputs;
        double** testTargets;
        int nTests;

        void randomize_patterns();
        void init_weights();
        void one_epoch();
};

#endif // NETWORK_H
