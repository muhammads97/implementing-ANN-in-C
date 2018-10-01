#include "network.h"

network::network(){
}

void network::init_network(int nInput, int nOutput, int nHidden, int nPattern){
    this->nInput = nInput;
    this->nOutput = nOutput;
    this->nHidden = nHidden;
    this->nPattern = nPattern;
}

void network::set_training(double** input, double** target){
    this->input = input;
    this->target = target;
}

void network::set_constants(double eta, double alpha, double smallwt){
    this->alpha = alpha;
    this->eta = eta;
    this->smallwt = smallwt;
}

void network::set_nEpoch(int nEpoch){
    this->nEpoch = nEpoch;
}

void network::init_weights(){
    this->weightIH = new double*[nInput +1];
    this->DeltaWeightIH = new double*[nInput +1];

    /* initialize WeightIH and DeltaWeightIH */
    for( i = 0 ; i < this->nInput + 1 ; i++ ) {
        weightIH[i] = new double[nHidden +1];
        DeltaWeightIH[i] = new double[nHidden + 1];
        for( j = 0 ; j < this->nHidden ; j++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
            weightIH[i][j] = 2.0 * (rando() - 0.5) * smallwt;
            //weightIH[i][j] = rando();
        }
    }
    this->weightHO = new double*[nHidden +1];
    this->DeltaWeightHO = new double*[nHidden +1];
    /* initialize WeightHO and DeltaWeightHO */
    for( j = 0 ; j < nHidden + 1; j++ ) {
        weightHO[j] = new double[nOutput +1];
        DeltaWeightHO[j] = new double[nOutput +1];
        for( k = 0 ; k < nOutput ; k ++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
            weightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
            //weightHO[j][k] = rando();
        }
    }
    //printf("%f \n", weightIH[nInput][nHidden]);
}

void network::randomize_patterns(){
    ranpat = new int[nPattern];
    for( p = 0 ; p < nPattern ; p++ ) {    /* randomize order of training patterns */
        ranpat[p] = p ;
    }
    for( p = 0 ; p < nPattern ; p++) {
        np = p + rando() * ( nPattern + 1 - p ) ;
        op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
    }
}

void network::one_epoch(){
    this->randomize_patterns();
    Error = 0.0 ;
    sumH = new double*[nPattern];
    hidden = new double*[nPattern];
    sumO = new double*[nPattern];
    output = new double*[nPattern];
    deltaO = new double[nOutput];
    sumDOW = new double[nHidden];
    deltaH = new double[nHidden];
    for( np = 0 ; np < nPattern ; np++ ) {    /* repeat for all the training patterns */
        p = ranpat[np];
        //printf("==%d=%d=\n", p, ranpat[nPattern - 1]);
        InputHandle jj;
        //jj.printArr(input[p], nInput);*/
        sumH[p] = new double[nHidden];
        hidden[p] = new double[nHidden];
        sumO[p] = new double[nOutput];
        output[p] = new double[nOutput];
        for( j = 0 ; j < nHidden; j++ ) {    /* compute hidden unit activations */
            sumH[p][j] = weightIH[0][j];
            for(i = 0 ; i < nInput; i++ ) {
                sumH[p][j] += input[p][i] * weightIH[i + 1][j] ;
            }
            hidden[p][j] = 1.0/(1.0 + exp(-sumH[p][j])) ;
           // hidden[p][j] = fmax(0, sumH[p][j]);
        }

        //jj.printArr(hidden[p], nHidden);
        for( k = 0 ; k < nOutput ; k++ ) {    /* compute output unit activations and errors */
            sumO[p][k] = weightHO[0][k] ;
            for( j = 0 ; j < nHidden ; j++ ) {
                sumO[p][k] += hidden[p][j] * weightHO[j + 1][k] ;
            }
            //printf("%f\n", sumO[p][k]);
            output[p][k] = 1.0/(1.0 + exp(-sumO[p][k])) ;   /* Sigmoidal Outputs */
            Error -= ( target[p][k] * log( output[p][k] ) + ( 1.0 - target[p][k] ) * log( 1.0 - output[p][k] ) ) ;    //Cross-Entropy Error
            deltaO[k] = target[p][k] - output[p][k];     //Sigmoidal Outputs, Cross-Entropy Error
        }
        //jj.printArr(output[p], 10);
        for(j = 0 ; j < nHidden ; j++) {    /* 'back-propagate' errors to hidden layer */
            sumDOW[j] = 0.0 ;
            for(k = 0; k < nOutput; k++ ) {
                sumDOW[j] += weightHO[j + 1][k] * deltaO[k] ;
            }
            deltaH[j] = sumDOW[j] * hidden[p][j] * (1.0 - hidden[p][j]) ;
        }
        for( j = 0 ; j < nHidden ; j++ ) {     /* update weights WeightIH */
            DeltaWeightIH[0][j] = eta * deltaH[j] + alpha * DeltaWeightIH[0][j] ;
            weightIH[0][j] += DeltaWeightIH[0][j] ;
            for( i = 0 ; i < nInput ; i++ ) {
                DeltaWeightIH[i + 1][j] = eta * input[p][i] * deltaH[j] + alpha * DeltaWeightIH[i + 1][j];
                weightIH[i + 1][j] += DeltaWeightIH[i + 1][j] ;
            }
        }
        for( k = 0 ; k < nOutput ; k ++ ) {    /* update weights WeightHO */
            DeltaWeightHO[0][k] = eta * deltaO[k] + alpha * DeltaWeightHO[0][k] ;
            weightHO[0][k] += DeltaWeightHO[0][k] ;
            for( j = 0 ; j < nHidden ; j++ ) {
                DeltaWeightHO[j + 1][k] = eta * hidden[p][j] * deltaO[k] + alpha * DeltaWeightHO[j + 1][k] ;
                weightHO[j + 1][k] += DeltaWeightHO[j + 1][k] ;
            }
        }
    }
    for(int x = 0; x < nPattern; x ++){
        delete [] sumH[x];
        delete [] hidden[x];
        delete [] sumO[x];
        delete [] output[x];
    }
    delete [] sumH;
    delete [] hidden;
    delete [] sumO;
    delete [] output;
    delete [] deltaO;
    delete [] sumDOW;
    delete [] deltaH;
}

void network::start_training(){
    init_weights();

    for(epoch = 0; epoch < nEpoch; epoch++){
        one_epoch();
        fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;
        if( Error < 0.2 ) break ;
        if(Error < 1.0) eta = 0.3;
        if(Error < 0.5) eta = 0.5;
        if(Error < 0.4) eta = 0.7;
    }
}

void network::store_weights(){
    std::ofstream file;
    file.open("F:\\implementing ANN\\handwritten digits dataset\\weightIH.res");
    for(i = 0; i < nInput + 1; i ++){
        for(j = 0; j < nHidden; j++){
            file << weightIH[i][j] << ", ";
        }
        file << weightIH[i][nHidden] << std::endl;
    }
    file.close();
    file.open("F:\\implementing ANN\\handwritten digits dataset\\weightHO.res");
    for(i = 0; i < nHidden + 1; i ++){
        for(j = 0; j < nOutput; j++){
            file << weightHO[i][j] << ", ";
        }
        file << weightHO[i][nOutput] << std::endl;
    }
    file.close();
}

void network::set_test(double** testInputs, double** testTargets, int nTests){
    this-> testInputs = testInputs;
    this->testTargets = testTargets;
    this->nTests = nTests;
}

void network::run_test(){
    Error = 0.0 ;
    sumH = new double*[nTests +1];
    hidden = new double*[nTests +1];
    sumO = new double*[nTests +1];
    testOutputs = new double*[nTests +1];
    for( p = 0 ; p < nTests ; p++ ) {    /* repeat for all the training patterns */
        sumH[p] = new double[nHidden];
        hidden[p] = new double[nHidden];
        sumO[p] = new double[nOutput];
        testOutputs[p] = new double[nOutput];
        for( j = 0 ; j < nHidden ; j++ ) {    /* compute hidden unit activations */
            sumH[p][j] = weightIH[0][j] ;
            for( i = 0 ; i < nInput ; i++ ) {
                sumH[p][j] += testInputs[p][i] * weightIH[i + 1][j] ;
            }
            hidden[p][j] = 1.0/(1.0 + exp(-sumH[p][j])) ;
        }
        for( k = 0 ; k < nOutput ; k++ ) {    /* compute output unit activations and errors */
            sumO[p][k] = weightHO[0][k] ;
            for( j = 0 ; j < nHidden ; j++ ) {
                sumO[p][k] += hidden[p][j] * weightHO[j + 1][k] ;
            }
            testOutputs[p][k] = 1.0/(1.0 + exp(-sumO[p][k])) ;   /* Sigmoidal Outputs */
            Error -= (testTargets[p][k] * log( testOutputs[p][k]) + (1.0 - testTargets[p][k]) * log(1.0 - testOutputs[p][k]));    //Cross-Entropy Error
        }
    }
    printf("test ended with ERROR : %f\n", Error);
}

void network::print_res(){
    printf("\n\nTest Results\n");
    for(int q = 0; q < nTests; q++){
        for(int r = 0; r < nOutput; r++){
            printf("%f ,,, ", testOutputs[q][r]);
        }
        printf("\n");
    }
}
