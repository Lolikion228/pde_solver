#include "cstring"

const int MAX_DIM = 8; 

class Point{
    private:
        int dim;
        double cords[MAX_DIM];

    public:
        Point(int _dim): dim(_dim) {};

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

