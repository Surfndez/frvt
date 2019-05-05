#include "Inverse.h"

#include <torch/script.h>

at::Tensor InverseMatrix(const at::Tensor &matTensor)
{
    int mat[3][3], i, j;
	
    // Enter elements of matrix row wise
	for(i = 0; i < 3; i++) {
		for(j = 0; j < 3; j++) {
            mat[i][j] = matTensor[i][j].item<float>();
        }
    }

    //finding determinant
    float determinant = 0;
	for(i = 0; i < 3; i++)
		determinant = determinant + (mat[0][i] * (mat[1][(i+1)%3] * mat[2][(i+2)%3] - mat[1][(i+2)%3] * mat[2][(i+1)%3]));

    // Inverse matrix
    std::vector<float> inverseMat;
	for(i = 0; i < 3; i++) {
		for(j = 0; j < 3; j++) {
			auto v = ((mat[(j+1)%3][(i+1)%3] * mat[(j+2)%3][(i+2)%3]) - (mat[(j+1)%3][(i+2)%3] * mat[(j+2)%3][(i+1)%3]))/ determinant;
            inverseMat.push_back(v);
        }
    }

    auto newMatTensor = torch::tensor(inverseMat, torch::requires_grad(false).dtype(torch::kFloat32)).view({3, 3});

    return newMatTensor;
}
