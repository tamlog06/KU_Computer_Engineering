#include <stdio.h>

// calculate nCk
float comb(float n, int k){
    if(k < 0) return 0;
    if(n == k || k == 0) return 1;

    float result = 1;
    for (int i = 0; i <= k-1; i++) {
        result *= (n-i);
    }
    for (int i = 1; i <= k; i++) {
        result /= i;
    }
    return result;
}

float calc_a(float n, int k){
    return comb(n, k) * comb((n+k-1)/2, n);
}

void test_print_a(int n){
    for (int i = 0; i <= n; i++) {
        printf("%d, %d\n", n, i);
        printf("%f\n", calc_a(n, i));
    }
}

int main()
{
    float n = -4.5;
    int k = 0;

    printf("%f %d\n", n, k);
    printf("%f\n", comb(n, k));

    test_print_a(5);
    test_print_a(10);
    return 0;
}