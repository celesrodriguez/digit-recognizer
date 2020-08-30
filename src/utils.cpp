
#include "utils.h"

#include <string>
#include <vector>
#include <iterator>
	// back_insert_iterator, back_inserter
#include <algorithm>
	// find, find_if, find_if_not, copy, count
#include <fstream>
	// ifstream, ofstream

#include "types.h"
#include <Eigen/Core>

#include <iostream>

inline bool is_part_of_number(char c) {
    static std::string pattern = "0123456789.";
    std::string::const_iterator it = std::find(pattern.begin(), pattern.end(), c);
    return ( it != pattern.end() );
}

void split(std::string::const_iterator begin_it, std::string::const_iterator end_it, std::back_insert_iterator< std::vector<double> > out_it) {
    typedef std::string::const_iterator iter;
    while (begin_it != end_it) {
        iter it = std::find_if(begin_it, end_it, is_part_of_number);
        iter jt = std::find_if_not(it, end_it, is_part_of_number);
        if (it != jt) {
            *out_it++ = std::stod(std::string(it, jt));
        }
        begin_it = jt;
    }
}

void read_train_data(Matrix& mdata, Matrix& mlabels, const char *filename) {
    std::ifstream ifs(filename);

    // defino la cantidad de columnas
    mdata.resize(0,COLS_TRAIN-1);
    mlabels.resize(0,1);
    std::string s;
    std::vector<double> v;
    std::getline(ifs, s); // descarto primera linea con headings (asumo formato [label,pixel0..pixel783]
    int i = 0;
    while ( std::getline(ifs, s) ) {
        // agrego una nueva fila
        mdata.conservativeResize(mdata.rows()+1, Eigen::NoChange);
        mlabels.conservativeResize(mlabels.rows()+1, Eigen::NoChange);

        // reuso funcion tp1
        split(s.begin(), s.end(), std::back_inserter(v)); // reuso funcion tp1

        // lleno la nueva fila
        mlabels(i,0) = v[0];
        for (int j = 1; j < COLS_TRAIN; ++j) {
            mdata(i,j-1) = v[j];
        }

        v.clear();
        ++i;
    }
}

void read_test_data(Matrix& mtest, const char *filename) {
    std::ifstream ifs(filename);

    // defino la cantidad de columnas
    mtest.resize(0,COLS_TEST);

    std::string s;
    std::vector<double> v;
    std::getline(ifs, s); // descarto primera linea con headings (asumo formato [pixel0..pixel783]
    int i = 0;
    while ( std::getline(ifs, s) ) {
        // agrego una nueva fila
        mtest.conservativeResize(mtest.rows()+1, Eigen::NoChange);

        // reuso funcion tp1
        split(s.begin(), s.end(), std::back_inserter(v)); // reuso funcion tp1

        // lleno la nueva fila
        for (int j = 0; j < COLS_TEST; ++j) {
            mtest(i,j) = v[j];
        }

        v.clear();
        ++i;
    }
}

void read_data(Matrix& mdata, Matrix& mlabels, char *filename) {
    std::ifstream ifs(filename);

    // defino la cantidad de columnas
    mdata.resize(0,COLS_TRAIN-1);

    std::string s;
    std::vector<double> v;
    std::getline(ifs, s); // descarto primera linea con headings
    unsigned int commas = std::count(s.begin(), s.end(), ',');
    bool flag_labels = 0;
    if ( commas == COLS_TRAIN-1 ) {
        // el archivo vino con labels
        flag_labels = 1;
        mlabels.resize(0,1); // defino la cantidad de columnas
    }

    int i = 0;
    while ( std::getline(ifs, s) ) {
        // reuso funcion tp1
        split(s.begin(), s.end(), std::back_inserter(v)); // reuso funcion tp1

        // agrego una nueva fila
        mdata.conservativeResize(mdata.rows()+1, Eigen::NoChange);

        if (flag_labels) {
            // agrego y lleno una fila en mlabels
            mlabels.conservativeResize(mlabels.rows()+1, Eigen::NoChange);
            mlabels(i,0) = v[0];
            v.erase(v.begin()); // saco la label asi no entra en mdata
        }

        // lleno la nueva fila
        for (int j = 0; j < COLS_TRAIN; ++j) {
            mdata(i,j) = v[j];
        }

        v.clear();
        ++i;
    }
}

void write_matrix_data(const Matrix& m, char *filename) {
    static Eigen::IOFormat tpFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n", "", "", "", "\n"); // armamos una configuracion de printeo amigable para el tp
    std::ofstream ofs(filename);
    ofs << m.format(tpFormat); // Eigen se encarga de printear
}

void write_vector_data(const Vector& v, char *filename) {
    std::ofstream ofs(filename);
    ofs << "ImageId,Label\n";
    unsigned int size = v.size();
    for(unsigned int i = 0; i < size; ++i) {
        ofs << i+1 << "," << v(i) << "\n";
    }
}

double get_accuracy_score(const Matrix& test, const Vector& res) {
    double hits = 0.0;
    double total = 0.0;
    unsigned int size = res.size();
    for(unsigned int i = 0; i < size; ++i) {
        if ( test(i,0) == res(i) ) {
            ++hits;
        }
        ++total;
    }

    return hits / total;
}

