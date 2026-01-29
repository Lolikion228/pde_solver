#include "diff.h"
#include "utils.hpp"

double df( double f(double),  double x, double h){
    return (f(x + h) - f(x - h)) / (2 * h);
}


double ddf( double f(double),  double x, double h){
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}
