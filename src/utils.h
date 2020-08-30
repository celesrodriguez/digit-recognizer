#ifndef GUARD_utils_h
#define GUARD_utils_h

#include <string>
#include <vector>
#include <iterator>
	// back_insert_iterator, back_inserter
#include <algorithm>
	// find, find_if, find_if_not, copy
#include <fstream>
	// ifstream, ofstream
#include "types.h"


const int COLS_TRAIN = 785; // [label, pixel0..pixel783]
const int COLS_TEST = 784; // [pixel0..pixel783]
const int ROWS_TRAIN = 42000;

inline bool is_part_of_number(char c);

void split(std::string::const_iterator begin_it, std::string::const_iterator end_it, std::back_insert_iterator< std::vector<double> > out_it);

void read_train_data(Matrix& mdata, Matrix& mlabels, const char *filename);

void read_test_data(Matrix& mtest, const char *filename);

void read_data(Matrix& mdata, Matrix& mlabels, char *filename);

void write_matrix_data(const Matrix& m, char *filename);
void write_vector_data(const Vector& v, char *filename);

double get_accuracy_score(const Matrix& test, const Vector& res);

#endif

