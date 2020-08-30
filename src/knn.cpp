#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <vector>
#include <algorithm>
#define DIGITS 9
using namespace std;


KNNClassifier::KNNClassifier(unsigned int k_neighbors){
	this->k_neighbors = k_neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y){
	this->data = X;
	this->dataLabels = y;
}

Vector KNNClassifier::predict(Matrix X){
    //Creamos vector columna a devolver
    Vector ret = Vector(X.rows());

    for (unsigned int i = 0; i < X.rows(); ++i){
        ret(i) = predictRow(X.row(i));
    }
  
    return ret;
}

unsigned int KNNClassifier::predictRow(RowVector v){
	//en este vector voy a guardar las labels y las normas del vector con esa label
    std::vector<std::pair<unsigned int,double>> distances(data.rows());
    
    for (unsigned int i = 0; i < data.rows(); i++) {
    	//calculo la distancia del vector v a la filai de nuestra matriz data
        RowVector tmp = v - data.row(i);
        double norm = tmp.norm();
        distances[i] = std::make_pair(dataLabels(i,0),norm);
    }

    //hago sort de índices de acuerdo a la norma
	std::sort(distances.begin(), distances.end(), [](const std::pair<unsigned int,double> &left, const std::pair<unsigned int,double> &right) {
    	return left.second < right.second;});

	return getMostVoted(distances);
}

unsigned int KNNClassifier::getMostVoted(std::vector< std::pair < unsigned int,double> > & normsAndLabels){
	std::vector<unsigned int> labels(DIGITS+1, 0);
	for (unsigned int i = 0; i < k_neighbors; ++i) {
		labels[normsAndLabels[i].first] += 1;
	}
	unsigned int max = labels[0]; 
	unsigned int mostVoted = 0;
	for (unsigned int i = 1; i < DIGITS; ++i) {
		if (labels[i] > max){
			max = labels[i];
			mostVoted = i;
		} 
	}
	return mostVoted;
}
