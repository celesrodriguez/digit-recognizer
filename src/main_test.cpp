//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"
#include "knn.h"
#include "utils.h"
using namespace std;

//Test método de potencia para matriz diagonal
void TestMetodoPotenciaDiagonal(){
    cout << "/////////////////////////////////////////////////////////" << endl;
    cout << "Método de la potencia para matriz diagonal" << endl;
    Matrix X(3,3);
    X.row(0) << 2, 0, 0;
    X.row(1) << 0, 3, 0;
    X.row(2) << 0, 0, 1; 

    pair<double, Vector> res;
    double tolerance_epsilon = 0.001;
    res = power_iteration(X, 1000);

    cout << "Primer autovalor de matriz 3x3:" << res.first << endl;
    assert(abs(res.first - 3) < tolerance_epsilon);
    cout << "///////////   TEST SUPERADO   /////////////////////////" << endl;
}

//Test primeros autovalores y autovectores para matriz diagonal
void TestMetodoDeflacionDiagonal(){
    cout << "/////////////////////////////////////////////////////////" << endl;
    cout << "Método de deflación para matriz diagonal" << endl;
    Matrix X(3,3);
    X.row(0) << 2, 0, 0;
    X.row(1) << 0, 3, 0;
    X.row(2) << 0, 0, 1;

    pair<Vector, Matrix> res;
    res = get_first_eigenvalues(X, 3,5000);
    Vector autovalores = res.first;
    cout << "Primer autovalor de matriz 3x3:" << autovalores(0) << endl;
    cout << "Segundo autovalor de matriz 3x3:" << autovalores(1) << endl;
    cout << "Tercer autovalor de matriz 3x3:" << autovalores(2) << endl;

    double tolerance_epsilon = 0.001;
    
    assert(abs(autovalores(0) - 3) < tolerance_epsilon);
    assert(abs(autovalores(1) - 2) < tolerance_epsilon);
    assert(abs(autovalores(2) - 1) < tolerance_epsilon);
    cout << "///////////   TEST SUPERADO   /////////////////////////" << endl;
}

int main(int argc, char** argv){
    TestMetodoPotenciaDiagonal();
    TestMetodoDeflacionDiagonal();
    return 0;
}
