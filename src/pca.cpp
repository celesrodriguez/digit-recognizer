#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) {
  this->alpha_componentes = n_components;
}

void PCA::fit(Matrix X) {
  RowVector mu = X.colwise().sum();
  mu = mu / X.rows();

  for (int i = 0; i < X.rows(); i++) {
    X.row(i) = X.row(i) - mu;
  }  

  Matrix covariance = X.transpose() * X;
  covariance = covariance / (X.rows()-1);
  this->V = get_first_eigenvalues(covariance, alpha_componentes).second;
}

Matrix PCA::transform(Matrix X){
	return X * V;
}
