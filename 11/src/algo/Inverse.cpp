#include "Inverse.h"

#include <torch/script.h>

at::Tensor InverseMatrix(at::Tensor &mat)
{
    int size = int(mat.sizes()[0]);

    auto determinant = torch::tensor(0, torch::requires_grad(false).dtype(torch::kFloat32));
	//finding determinant
	for(int i = 0; i < 3; i++)
		determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));

    auto newMat = at::zeros(size);

    for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			newMat[i][j] = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]))/ determinant;
	    }
    }

    return newMat;
}