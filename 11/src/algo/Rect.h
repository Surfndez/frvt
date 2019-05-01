#ifndef RECT_H_
#define RECT_H_

namespace FRVT_11 {
    struct Rect {
    Rect(x1, y1, x2, y2): x1(x1), y1(y1), x2(x2), y2(y2);
    int x1, y1, x2, y2;
};
}

#endif /* RECT_H_ */
