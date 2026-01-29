#include "utils.h"

double cut(double f(Point), Point x, int i, double t){
    Point e(x.dim, i, t);
    return f(x + e);
}