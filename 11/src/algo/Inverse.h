#ifndef INVERSE_H_
#define INVERSE_H_

namespace at {
    class Tensor;
}

at::Tensor InverseMatrix(const at::Tensor &t);

#endif /* INVERSE_H_ */