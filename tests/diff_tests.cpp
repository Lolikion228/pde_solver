#include "diff_tests.h"

#define func std::function<double(double)>


void test_pipline1(func f1, func df1, func ddf1, double a, double b, int n){
    double h = 0.1;
    double x, df_delta, ddf_delta;
    for(int i=0; i<4; ++i){
        x = a;
        printf("h = %e\n", h);

        std::cout << "x      ";
        std::cout << "df_delta   ";
        std::cout << "ddf_delta\n";

        for(int j=0; j<n; ++j){
            x += (b - a) / n;
            df_delta = abs(df1(x) - df(f1, x, h));
            ddf_delta = abs(ddf1(x) - ddf(f1, x, h));
            printf("%3.4f %3.4e %3.4e\n", x, df_delta, ddf_delta);
        }

        h /= 10;
        std::cout << "\n";
    }
}

void diff_test1(){
    func f1 = [](double x){
        return 5 * pow(x, 7) + 4 * pow(x, 3) + x;};

    func df1 = [](double x){
        return 35 * pow(x, 6) + 12 * pow(x, 2) + 1;};

    func ddf1 = [](double x){
        return 210 * pow(x, 5) + 24 * pow(x, 1);};

    double a = -2;
    double b = 2;
    int n = 6;

    test_pipline1(f1, df1, ddf1, a, b, n);
}


void diff_test2(){
    func f1 = [](double x){
        return sin(x) + exp(2 * x) + cos(x) * cos(x);};

    func df1 = [](double x){
        return cos(x) + 2 * exp(2 * x) - 2 * cos(x) * sin(x);};

    func ddf1 = [](double x){
        return -sin(x) + 4 * exp(2 * x) - 2 * cos(2 * x);};

    double a = -5;
    double b = 5;
    int n = 6;

    test_pipline1(f1, df1, ddf1, a, b, n);
}


