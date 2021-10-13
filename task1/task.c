#include <stdio.h>
#include <math.h>

// calcurate nCk
double comb(double n, int k){
    // 例外処理
    if(k < 0) return 0;
    if(n == k || k == 0) return 1;

    double result = 1;
    for (int i = 0; i <= k-1; i++) {
        result *= (n-i);
    }
    for (int i = 1; i <= k; i++) {
        result /= i;
    }
    return result;
}

// calcurate a
double calc_a(double n) {
    return n;
}







// print a
void test_print_a(int n){
    for (int i = 0; i <= n; i++) {
        printf("%d, %d\n", n, i);
        printf("%f\n", calc_a(n, i));
    }
}

double Pn_x(double n, double x) {
    // double result = pow(2, n);
    double result = 1;
    double sum = 0;
    for (int k=0; k <= n; k++){
        sum += calc_a(n, k);
        sum *= pow(x, k);
    }
    result *= sum;
    return result;
}

// dichotomy
void dichotomy(double n){
    const double lower = -1;
    const double upper = 1;
    double x = lower - (upper - lower) / n ;

    for (int i=0; i<=n; i++) {
        x += (upper - lower) / n;
        printf("x = %lf, P(x) = %lf\n", x, Pn_x(n, x));
    }
}

int main() {
    double n = -6;
    int k = 3;

    printf("%f %d\n", n, k);
    printf("%f\n", comb(n, k));

    // test_print_a(5);
    // test_print_a(10);

    dichotomy(10);
    return 0;
}