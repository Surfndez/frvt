#ifndef TIMEMEASUREMENT_H_
#define TIMEMEASUREMENT_H_

#include <string>
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

std::string ValidateNumThreads();

#endif /* TIMEMEASUREMENT_H_ */
