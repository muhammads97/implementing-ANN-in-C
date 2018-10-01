#include <iostream>
#include "InputHandle.h"
#include "network.h"

using namespace std;

int main()
{
    std::string n = "F:\\implementing ANN\\handwritten digits dataset\\optdigits.tra";
    std::string n2 = "F:\\implementing ANN\\handwritten digits dataset\\optdigits.tes";
    InputHandle i;
    InputHandle test;
    test.setLocation(n2);
    test.readFile();
    test.setInputs();
    i.setLocation(n);
    i.readFile();
    i.setInputs();
    network net;
    net.init_network(i.get_nInputs(), 10, i.get_nInputs(), i.get_nPatterns());
    net.set_training(i.get_inputs(), i.get_targets());
    net.set_constants(0.1, 0.7, 0.5);
    net.set_nEpoch(100000);
    net.start_training();
    net.store_weights();
    net.set_test(test.get_inputs(), test.get_targets(), test.get_nPatterns());
    net.run_test();
    net.print_res();
    return 0;
}
