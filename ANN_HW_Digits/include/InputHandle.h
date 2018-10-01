#ifndef INPUTHANDLE_H
#define INPUTHANDLE_H
#include <string>
#include <list>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

using namespace std;

class InputHandle
{
    public:
        InputHandle();
        void setLocation(std::string file);
        std::string getLocation();
        void readFile();
        void init_out();
        void setInputs();
        double** get_inputs();
        double** get_targets();
        int get_nInputs();
        int get_nPatterns();
        void printArr(double* arr, int arrLen);
    protected:
    private:
        std::string fileLocation;
        list<string> read;
        int nPatterns;
        int nInputs;
        double** inputs;
        double** targets;
};

void split (int* arr, std::string str, int arrLen);
void printArr(int *arr, int arrLen);


#endif // INPUTHANDLE_H
