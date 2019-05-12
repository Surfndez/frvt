#ifndef TIMEMEASUREMENT_H_
#define TIMEMEASUREMENT_H_

#include <chrono>

namespace FRVT_11 {
    class TimeMeasurement {
public:
    TimeMeasurement();
    void Test();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
};
}

#endif /* TIMEMEASUREMENT_H_ */
