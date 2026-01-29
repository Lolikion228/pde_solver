#include "diff/diff.h"
#include "iostream"

double f1(double x){
    return x*x;
}

int main(){
    std::cout << df(f1, 3, 1e-3) << "\n";
}