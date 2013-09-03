#include <stdio.h>
#include <iostream>
#include <math.h>
#include "common.h"
#include "lasso.cpp"
#include "shared.cpp"

class Shotgun {
		double lambda;
		int N;
		int d;
		shotgun_data sd;
		double threshold;
		int K;
		int maxIter;
		int useOffset;
		int verbose;
		int numThreads;

		void prep_shape(int N, int d) {
			this->N = N;
			this->d = d;

			sd.ny = N;
			sd.nx = d;
			
			// Make sure that matrix is empty:
			sd.A_cols.clear();
			sd.A_rows.clear();
			
			sd.A_rows.resize(N);
			sd.A_cols.resize(d);

		}

		void add_nonzero(int row, int col, double val) {
				sd.A_cols[col].add(row, val);
				sd.A_rows[row].add(col, val);
		}

	public:
		Shotgun() {
			useOffset = 1;		
			threshold = 1;
			K = 0;
			maxIter = 5e6;
			verbose = 0;
			numThreads = 4;
		}
	
		void set_A(double* data, int N, int d, long* strides) {
	
			bool columnMajor = (strides[0] < strides[1]);

			// Set N, set d, allocate memory, etc:
			prep_shape(N, d);

			int i = 0;
			if (columnMajor) {
				for (int col=0; col < d; col++) {
					for (int row=0; row < N; row++) {
						double val = data[i++];
						if (val != 0) {
							add_nonzero(row, col, val);
						}
					}
				}
			} else {
				for (int row=0; row < N; row++) {
					for (int col=0; col < d; col++) {
						double val = data[i++];
						if (val != 0) {
							add_nonzero(row, col, val);
						}
					}
				}
			}
		}

		void set_A_sparse(double* data, int* indices, int nnz, int N, int d) {
			// Assumes csc sparse matrix format

			// Set N, set d, allocate memory, etc:
			prep_shape(N, d);

			int i = 0;
			int col = 0;
			int last_row = 0;
			while (i < nnz) {
				int row = indices[i];
				if (row < last_row) 
					col++;
				add_nonzero(row, col, data[i++]);	
				last_row = row;
			}
		}

		void set_y(double* data, int N) {
			// Make sure that vector is empty:
			sd.y.clear();

			sd.y.reserve(N);
			for (int e=0; e < N; e++)
				sd.y.push_back(data[e]);
		}

		void set_lambda(double value) {
			lambda = value;
		}
	
		void set_threshold(double value) {
			threshold = value;
		}

		void set_use_offset(int value) {
			useOffset = value; 
		}

		void set_num_threads(int value) {
			numThreads = value;
		}

		void run(double* result) {
			if (numThreads > 0) {
				omp_set_num_threads(numThreads);
			}
			solveLasso(&sd, lambda, K, threshold, maxIter, useOffset, verbose);
			for (int f = 0; f < d; f++)
				result[f] = sd.x[f];
			result[d] = sd.b;
		}

};

extern "C" {
	Shotgun* Shotgun_new() { return new Shotgun(); }

	void Shotgun_set_A(Shotgun* s, double* data, int N, int d, long* strides) {
		s->set_A(data, N, d, strides);
	}

	void Shotgun_set_A_sparse(Shotgun* s, double* data, int* indices, int nnz, int N, int d) {
		s->set_A_sparse(data, indices, nnz, N, d);
	}

	void Shotgun_set_y(Shotgun* s, double* data, int length) {	
		s->set_y(data, length);
	}

	void Shotgun_set_lambda(Shotgun* s, double value) {
		s->set_lambda(value);
	}

	void Shotgun_set_threshold(Shotgun* s, double value) {
		s->set_threshold(value);
	}

	void Shotgun_set_use_offset(Shotgun* s, int value) {
		s->set_use_offset(value);
	}

	void Shotgun_set_num_threads(Shotgun* s, int value) {
		s->set_num_threads(value);
	}

	void Shotgun_run(Shotgun* s, double* result, long length) {
		s->run(result);
	}
}

