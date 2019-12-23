#ifndef RECT_H_
#define RECT_H_

#include <stdexcept>

namespace FRVT_11 {
    template <typename Type>
    struct RectTemplate {
    
    RectTemplate(Type x1, Type y1, Type x2, Type y2, float score): x1(x1), y1(y1), x2(x2), y2(y2), score(score) {}
    
    int Area()
    {
        return (x2 - x1 + 1) * (y2 - y1 + 1);
    }
    
    Type x1, y1, x2, y2;
    float score;

    Type operator[] (const int index)
    {
        if (index == 0) return x1;
        if (index == 1) return y1;
        if (index == 2) return x2;
        if (index == 3) return y2;
        throw std::runtime_error("wrong index to Rect");
    }
};

using Rect = RectTemplate<int>;
using RectF = RectTemplate<float>;

}

#endif /* RECT_H_ */
