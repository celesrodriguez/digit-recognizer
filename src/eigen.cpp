#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector v = Vector::Random(X.cols());
    Vector v_new;
    double eigenvalue;
    unsigned int i;
    for (i = 0; i < num_iter; ++i){
      v_new = X * v;
      v_new /= v_new.norm();
      if ((v_new - v).norm() < eps) break;

      v = v_new;
    }
    RowVector v_t = v.transpose();

    double scalar_mul =  v_t * v;
    eigenvalue = v_t * X * v;
    eigenvalue /= scalar_mul; // por algun motivo si lo pongo todo en la misma linea
    						 // me llora el compiler >.<

    // Hay que chequear si efectivamente es autovalor-autovector??? para eso es eps?
    return make_pair(eigenvalue, v);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A = X;
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    pair<double, Vector> currentEigenPair;
    Vector currentEigenvector;
    double currentEigenvalue;

    /* obtenemos los primeros num autovectores y autovalores
       con método de la deflación
    */
    for (unsigned int i = 0; i < num; i++) {
      //iésimo autovector y autovalor
      currentEigenPair = power_iteration(A, num_iter, epsilon);
      currentEigenvalue = currentEigenPair.first;
      currentEigenvector = currentEigenPair.second;
      //guardamos en matriz de autovectores y vector de autovalores respectivamente
      eigvalues(i) = currentEigenvalue;
      eigvectors.col(i) = currentEigenvector;
      //aplicamos el mismo paso para el i+1 autovalor con su autovector
      RowVector currentEigenvector_t = currentEigenvector.transpose();
      A = A - currentEigenvalue * currentEigenvector * currentEigenvector_t;
    }
    return make_pair(eigvalues, eigvectors);
}
