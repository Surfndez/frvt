#include "Rect.h"

using namespace FRVT_11;

int& Rect::operator[] (const int index)
{
    if (index == 0) return x1;
    if (index == 1) return y1;
    if (index == 2) return x2;
    if (index == 3) return y2;
    throw std::runtime_error("wrong index to Rect");
}
