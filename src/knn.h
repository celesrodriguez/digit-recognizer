#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int k_neighbors);
    void fit(Matrix X, Matrix Y);
    Vector predict(Matrix X);
    unsigned int predictRow(RowVector v);
    unsigned int getMostVoted(std::vector< std::pair <unsigned int,double> > & normsAndLabels);

private:
    //datos de entrenamiento (pixels)
    Matrix data;
    //labels
    Matrix dataLabels;
    //los n vecinos
	unsigned int k_neighbors;
};
