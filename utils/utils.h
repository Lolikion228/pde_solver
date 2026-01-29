#include "cstring"
#include <functional>

const int MAX_DIM = 8; 

class Point{
    public:
        int dim;

        double cords[MAX_DIM]{};

        Point(int _dim): dim(_dim) {};
        Point(int _dim, int i, double t): dim(_dim) {cords[i]=t;};

        Point(const Point& p){
            dim = p.dim;
            memcpy(cords, p.cords, sizeof(p.cords));
        };

        Point& operator += (const Point& p){
            for(int i=0; i<dim; ++i){
                cords[i] += p.cords[i];
            }
            return *this;
        }

        Point& operator + (const Point& p){
            Point p2 = p;
            return (p2 += p);
        }
};

double cut(double f(Point), Point x, int i, double t);