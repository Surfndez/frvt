#ifndef INVERSE_H_
#define INVERSE_H_

namespace at {
    class Tensor;
}

at::Tensor InverseMatrix(const at::Tensor &matTensor);

#endif /* INVERSE_H_ */