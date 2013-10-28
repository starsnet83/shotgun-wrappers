#include <stdio.h>
#include <iostream>
#include <math.h>
#include "common.h"

class Shotgun {
		double lambda;
		int N;
		int d;
		shotgun_data sd;
		double threshold;
		int maxIter;
		int useOffset;
		int verbose;
		int numThreads;

		double* xInitial;
		double offsetInitial;

		void prep_shape(unsigned int N, unsigned int d) {
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

		void add_nonzero(unsigned int row, unsigned int col, double val) {
			if (val == 0)
				return;
			sd.A_cols[col].add(row, val);
			sd.A_rows[row].add(col, val);
		}

	public:
		Shotgun() {
			useOffset = 1;		
			threshold = 1e-5;
			maxIter = 5e6;
			verbose = 0;
			numThreads = 1;
			xInitial = NULL;
			offsetInitial = NULL;
		}
	
		void set_A(double* data, int N, int d, long* strides) {
 	
			bool columnMajor = (strides[0] < strides[1]);

			// Set N, set d, allocate memory, etc:
			prep_shape(N, d);

			int i = 0;
			if (columnMajor) {
				for (int col=0; col < d; col++) {
					for (int row=0; row < N; row++)
						add_nonzero(row, col, data[i++]);
				}
			} else {
				for (int row=0; row < N; row++) {
					for (int col=0; col < d; col++)
						add_nonzero(row, col, data[i++]);
				}
			}
		}

		void set_A_sparse(double* data, unsigned int* indices, unsigned int* indptr, unsigned int nnz, unsigned int N, unsigned int d) {
			// Assumes csc sparse matrix format

			// Set N, set d, allocate memory, etc:
			prep_shape(N, d);

			unsigned int i = 0;
			unsigned int col = 0;
			unsigned int col_end_i = indptr[1];
			while (i < nnz) {
				unsigned int row = indices[i];
				if (i == col_end_i) {
					col++;
					col_end_i = indptr[col+1];
				}
				add_nonzero(row, col, data[i]);	
				i++;
			}
			return;
		}

		void set_y(double* data, int N) {
			// Make sure that vector is empty:
			sd.y.resize(N);
			for (int e=0; e < N; e++)
				sd.y[e] = data[e];
		}

		void set_lambda(double value) {
			lambda = value;
		}
	
		void set_threshold(double value) {
			this->threshold = double(value);
		}

		void set_use_offset(int value) {
			useOffset = value; 
		}

		void set_num_threads(int value) {
			numThreads = value;
		}

		void set_initial_conditions(double* x, double offset) {
			xInitial = x;
			offsetInitial = offset;
		}

		void run(double* result, std::string solver) {
			if (numThreads > 0) {
				omp_set_num_threads(numThreads);
			}
			if (solver == "lasso") {
				solveLasso(&sd, lambda, threshold, maxIter, useOffset, verbose, xInitial, offsetInitial);
			} else if (solver == "logreg") {
				compute_logreg(&sd, lambda, threshold, maxIter, useOffset, verbose, xInitial, offsetInitial);
			} else {
				assert(false);
			}
			for (int f = 0; f < d; f++)
				result[f] = sd.x[f];
			result[d] = sd.b;
		}

};

extern "C" {
	Shotgun* Shotgun_new() { 
		Shotgun* s = new Shotgun();
		return s; 
	}

	void Shotgun_set_A(Shotgun* s, double* data, int N, int d, long* strides) {
		s->set_A(data, N, d, strides);
	}

	void Shotgun_set_A_sparse(Shotgun* s, double* data, unsigned int* indices, unsigned int* indptr, unsigned int nnz, unsigned int N, unsigned int d) {
		s->set_A_sparse(data, indices, indptr, nnz, N, d);
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
		//s->set_num_threads(value);
	}

	void Shotgun_set_initial_conditions(Shotgun* s, double* x, double offset) {
		s->set_initial_conditions(x, offset);
	}

	void Shotgun_run_lasso(Shotgun* s, double* result) {
		s->run(result, "lasso");
	}
	
	void Shotgun_run_logreg(Shotgun* s, double* result) {
		s->run(result, "logreg");
	}

}

