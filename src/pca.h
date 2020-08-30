#pragma once
#include "types.h"

class PCA {
public:
	PCA(unsigned int n_components);
    void fit(Matrix X);
    Matrix transform(Matrix X);

private:
    unsigned int alpha_componentes;
    Matrix V;
};
