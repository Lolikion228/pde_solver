
// df/dx
double df(std::function<double(double)> f,  double x, double h);

// df/dx_i
double df( double f(Point), Point x, int i, double h);

// d^2f/d(x)^2
double ddf(std::function<double(double)> f,  double x, double h);

// d^2f/d(x_i)^2
double ddf( double f(Point), Point x, int i, double h);