#ifndef FACE_CLASSIFIER_H
#define FACE_CLASSIFIER_H

#include <vector>

namespace FRVT_11 {
    class FaceClassifier
    {
    public:
        bool classify(const std::vector<float>& features) const;
};
}

#endif /* FACE_CLASSIFIER_H */