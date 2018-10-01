#include "InputHandle.h"

void split (double* arr, std::string str, int arrLen){
    std::istringstream iss(str);
    char c; // dummy character for the colon
    iss >> arr[0];
    arr[0] = arr[0] / 16;
    for (int i = 1; i < arrLen - 1; i++){
        iss >> c >> arr[i];
        arr[i] = arr[i] / 16;
    }
    iss >> c >> arr[arrLen - 1];

}

InputHandle::InputHandle(){

}
std::string InputHandle::getLocation(){
    return this->fileLocation;
}
void InputHandle::setLocation(std:: string file){
    this->fileLocation = file;
}

void InputHandle::readFile(){
    list<string> read;
    std::ifstream input( this->fileLocation );
    std::string line;
    while(std::getline(input, line)){
        read.push_back(line);
    }
    this->nPatterns = read.size();
    this->read = read;
    input.close();
}

void InputHandle::setInputs(){
    this->inputs = new double*[nPatterns];
    this->targets = new double*[nPatterns];
    this->nInputs = 64;
    std::list<std::string>::iterator it = read.begin();
    int i = 0;
    while(it != read.end()){
        std::string s = *it;
        inputs[i] = new double[65];
        targets[i] = new double[10];
        memset(targets[i], 0, 10*sizeof(double));
        split(inputs[i], s, 65);
        //printf("%d\n", inputs[i][64]);
        targets[i][(int)inputs[i][64]] = 1.0;
        /*printf("==%d==\n", i);
        printArr(inputs[i], 64);
        printArr(targets[i], 10);*/
        std::advance(it, 1);
        i++;
    }
}

double** InputHandle::get_inputs(){
    return this->inputs;
}

int InputHandle::get_nInputs(){
    return this->nInputs;
}

double** InputHandle::get_targets(){
    return this->targets;
}

int InputHandle::get_nPatterns(){
    return this->nPatterns;
}

void InputHandle::printArr(double *arr, int arrLen){
    for(int i = 0; i < arrLen; i++){
        printf("%f ", arr[i]);
    }
    printf("\n");
}

