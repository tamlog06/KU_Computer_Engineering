#include <stdio.h>

/* calculate y = Ax, 
 * where
 *   A is (m rows) x (n columns) matrix in 2-D array
 *   x is 1-D vector with n elements
 *   y is 1-D vector with m elements
 */
void mul_mat_vec(int m, int n, double A[m][n], double x[n], double y[m])
{
    int i;
    for ( i = 0; i < m; i++ ) {
        y[i] = 0;
        int j;
        for ( j = 0; j < n; j++ ) {
            y[i] += A[i][j] * x[j];
        }
    }
}

/* print vector in 1-D array */
void print_vec(int m, double x[m])
{
    int i; 
    for ( i = 0; i < m - 1; i++ ) {
        printf(" %g,", x[i]);
    }
    /* last element */
    printf(" %g\n", x[i]);
}

/* print matix in 2-D (m x n) array */
void print_mat(int m, int n, double A[m][n])
{
    int i; 
    for ( i = 0; i < m; i++ ) {
        print_vec(n, A[i]);
    }
}

/* example to initialize matrix element */
void set_mat_example(int m, int n, double A[m][n])
{
    double v = 1.0;
    int i;
    for ( i = 0; i < m; i++ ) {
        int j;
        for ( j = 0; j < n; j++ ) {
            A[i][j] = v;
            v += 1.0;
        }
    }
}

int main(int ac, char *av[])
{
    double mat[3][2];
    set_mat_example(3, 2, mat);

    printf("mat = \n");
    print_mat(3, 2, mat);

    double a[2] = {-1, 2};

    printf("a = \n");
    print_vec(2, a);

    double b[3];
    mul_mat_vec(3, 2, mat, a, b);

    printf("b = \n");
    print_vec(3, b);

    return 0;
}