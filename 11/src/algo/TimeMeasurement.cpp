#include "TimeMeasurement.h"

#include <iostream>
#include <stdexcept>

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

std::string
ValidateNumThreads()
{
    const char* cmd = "top -H -b -n1 | grep validate11 | wc -l";
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}
