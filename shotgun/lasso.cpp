/*
   Copyright [2011] [Aapo Kyrola, Joseph Bradley, Danny Bickson, Carlos Guestrin / Carnegie Mellon University]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

// Optimization problem
//      \arg \min_{x,b} 0.5*||Ax + b1 - y||^2 + \lambda |x|_1
// where b1 is a scalar b times an all-ones vector

#include "common.h"

void initialize_feature(shotgun_data *sd, int feat_idx) {
	// Caches some values for the feat_idx'th column of A

	sparse_array& col = sd->A_cols[feat_idx];
	feature& feat = sd->feature_consts[feat_idx];
	sd->x[feat_idx] = 0.0;

	// Precompute norm of a feature:
	feat.covar = 0;
	for(int i=0; i<col.length(); i++) {
			feat.covar += sqr(col.values[i]);
	}
	feat.covar *= 2;
	
	// Precompute (A^T y)_i:
	feat.Ay_i = 0.0;
	feat.A1_i = 0.0;
	for(int i=0; i<col.length(); i++) {
		feat.Ay_i += col.values[i] * sd->y[col.idxs[i]];
		feat.A1_i += col.values[i];
	}
	feat.Ay_i *= 2;
	feat.A1_i *= 2;
}

void initialize(shotgun_data *sd, int useOffset, double* initial_x = NULL, double initial_offset = 0.0) {
	// Initializes problem, caching problem constants and initial conditions

	// Reserve space for weight vector and caching values:
	sd->x.resize(sd->nx);
	sd->feature_consts.reserve(sd->nx);
	sd->Ax.resize(sd->ny);

	// Initialize features (in parallel):
	#pragma omp for
	for(int i=0; i<sd->nx; i++) {
		initialize_feature(sd, i);
	}

	// Initialize offset:
	sd->b = 0.0;
	switch (useOffset) {
	case 0:
		break;
	case 1:
		// Initialize offset to initial condition if provided:
		if (initial_offset) 
			sd->b = initial_offset;
		break;
	default:
		assert(false); // Offset must be 0 or 1
	}

	// Initial weight vector values:
	if (initial_x) {
		for (int i = 0; i < sd->nx; i++) {
			if (initial_x[i] != 0) {
				double col_value = initial_x[i];
				sd->x[i] = col_value;
				sparse_array& col = sd->A_cols[i];
				int len = col.length();
				for (int j = 0; j < len; j++) 
					sd->Ax[col.idxs[j]] += col.values[j] * col_value;
			}
		}
	}

} 

valuetype_t soft_threshold(valuetype_t lambda, valuetype_t shootDiff) {
	// Soft threshold function for applying shrinkage:
	if (shootDiff > lambda) return lambda - shootDiff;
	if (shootDiff < -lambda) return -lambda - shootDiff;
	else return 0;
}

double shoot(shotgun_data *sd, int x_i, valuetype_t lambda) {
	
	// Get feature and current weight:
	feature& feat = sd->feature_consts[x_i];
	valuetype_t oldvalue = sd->x[x_i];

	// Return if feature is all zeros:
	if (feat.covar == 0.0) return 0.0;
	
	// Compute A_i^T*(Ax):
	valuetype_t AtAxj = 0.0;
	sparse_array& col = sd->A_cols[x_i];
	int len=col.length();
	for(int i=0; i<len; i++) {
			AtAxj += col.values[i] * sd->Ax[col.idxs[i]];
	}
	
	// New value without regularization:
	valuetype_t proposal =
		2 * AtAxj - feat.covar * oldvalue - feat.Ay_i + sd->b * feat.A1_i;

	// Apply shrinkage:
	valuetype_t newvalue = soft_threshold(lambda, proposal)/feat.covar;

	// Record difference:
	valuetype_t delta = (newvalue - oldvalue);
	
	// Update Ax:
	if (delta != 0.0) {
		for(int i=0; i<len; i++) {
		 sd->Ax.add(col.idxs[i], col.values[i] * delta);
		}
		sd->x[x_i] = newvalue;
	}

	return std::abs(delta);

}

void main_optimization_loop(shotgun_data *sd, double lambda, double threshold, int maxiter, int useOffset) {

	bool converged;
	double delta;
	double max_change;
	int itr = 0;

	lambda *= 2.0;  // Objective function is implemented without the leading 1/2

	// Loop until convergence:
	do {
	
		itr++;
		max_change = 0.0;
			
		// Update offset value (no regularization):
		if (useOffset) {
			double old_b = sd->b;
			sd->b = 0;
			for (int i = 0; i < sd->ny; ++i)
				sd->b += sd->y[i] - sd->Ax[i];
			sd->b /= sd->ny;
			if (old_b != sd->b)
				max_change = std::fabs(old_b - sd->b);
		}

		// Perfrom shoots on all coordinates in parallel (round robin):
		#pragma omp parallel for  
		for(int i=0; i<sd->nx; i++) {
			delta = shoot(sd, i, lambda);
			max_change = (max_change < delta ? delta : max_change);
		}

		// Convergence check:
		converged = (max_change <= threshold);

	} while (!converged && (itr < maxiter || maxiter == 0));

}

double solveLasso(shotgun_data *sd, double lambda, double threshold, int maxiter, int useOffset, double* initial_x, double initial_offset) {
	initialize(sd, useOffset, initial_x, initial_offset);
	main_optimization_loop(sd, lambda, threshold, maxiter, useOffset);
	return 0;
}

