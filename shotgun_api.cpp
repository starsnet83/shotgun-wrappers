#include <stdio.h>
#include <iostream>
#include <math.h>
#include "common.h"
#include "lasso.cpp"
#include "shared.cpp"

class Shotgun {
		double lambda;
		long N;
		long d;
		shotgun_data sd;
		double threshold;
		int K;
		int maxIter;
		int useOffset;
		int verbose;
		int numThreads;

	public:
		Shotgun() {
			useOffset = 1;		
			threshold = 1e-5;
			K = 0;
			maxIter = 1e6;
			verbose = 0;
			numThreads = 5;
		}
	
		void set_A(double* data, int dim, long* shape) {
			if (dim != 2) {
				std::cerr << "A matrix must be 2d\n";
				throw dim;
			}
			N = shape[0];
			d = shape[1];
			
			sd.ny = N;
			sd.nx = d;
			
			// Make sure that matrix is empty:
			sd.A_cols.clear();
			sd.A_rows.clear();
			
			sd.A_rows.resize(N);
			sd.A_cols.resize(d);

			long i = 0;
			for (int row=0; row < N; row++) {
				for (int col=0; col < d; col++) {
					double val = data[i++];
					if (val != 0) {
						sd.A_cols[col].add(row, val);
						sd.A_rows[row].add(col, val);
					}
				}
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

	void Shotgun_set_A(Shotgun* s, double* data, int dim, long* shape) {
		s->set_A(data, dim, shape);
	}

	void Shotgun_set_y(Shotgun* s, double* data, int length) {	
		s->set_y(data, length);
	}

	void Shotgun_set_lambda(Shotgun* s, double value) {
		s->set_lambda(value);
	}

	void Shotgun_run(Shotgun* s, double* result, long length) {
		s->run(result);
	}
}

