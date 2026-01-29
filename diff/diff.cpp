#include "diff.h"
#include "utils.hpp"
#include <functional>

double df(std::function<double(double)> f,  double x, double h){
    return (f(x + h) - f(x - h)) / (2 * h);
}

double df(double f(Point), Point x, int i, double h){
    std::function<double(double)> g = [x,i,f](double t){return cut(f, x, i, t);};
    return df(g, x.cords[i], h);
}

double ddf(std::function<double(double)> f,  double x, double h){
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}

double ddf(double f(Point), Point x, int i, double h){
    std::function<double(double)> g = [x,i,f](double t){ return cut(f, x, i, t);};
    return ddf(g, x.cords[i], h);
}