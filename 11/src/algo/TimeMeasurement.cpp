#include "TimeMeasurement.h"

#include <iostream>

using namespace FRVT_11;

TimeMeasurement::TimeMeasurement() : mStart(std::chrono::high_resolution_clock::now())
{
}

void
TimeMeasurement::Test()
{
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - mStart;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
}
