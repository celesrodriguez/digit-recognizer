//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include <iomanip>
#include "pca.h"
#include "eigen.h"
#include "knn.h"
#include "utils.h"
using namespace std;
#define KNN_method 0
#define PCA_method 1

int main(int argc, char** argv){
	//./tp2 -m <method> -i <train_set> -q <test_set> -o <classif> -k <k> -a <alfa>

  if (argc >= 9){
    string path = "../data/";
  	int method = atoi(argv[2]);
  	string train_set = argv[4];
  	string test_set = argv[6];
  	char *output = argv[8];
  	int k = 3;
  	int alfa = 34;

        if (argc >= 11) {
            k = atoi(argv[10]);
        }
        if (argc >= 13) {
            alfa = atoi(argv[12]);
        }

        Matrix data, dlabels;
        read_train_data(data, dlabels, (path + train_set).c_str());
        Matrix test, tlabels;
        read_test_data(test, (path + test_set).c_str());
        Vector res;

  	if (method == KNN_method){
  		//vamos a usar KNN
            KNNClassifier knn(k);
            knn.fit(data, dlabels);

            res = knn.predict(test);
            write_vector_data(res, output);
            std::cout << "Corro knn con k = " << k << std::endl;
  	}else{
  		//vamos a usar PCA + KNN
            PCA pca(alfa);
            pca.fit(data);
            KNNClassifier knn(k);
            knn.fit(pca.transform(data), dlabels);
            res = knn.predict(pca.transform(test));
            write_vector_data(res, output);
            std::cout << "Corro PCA y KNN con k = " << k << ", alfa = " << alfa << std::endl;
  	}

        if ( tlabels.size() != 0 ) {
            // el conjunto de test tiene labels
            double acc = get_accuracy_score(tlabels, res);
            std::cout << "accuracy: " << std::setprecision(4) << acc << std::endl;
        } else {
            std::cout << "accuracy: no hay labels en el conjunto de test!" << std::endl;
        }
        
  }

  return 0;
}
