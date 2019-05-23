#include "TestUtils.h"

std::vector<std::string>
ReadTestList(const std::string& list_path)
{
    std::ifstream is(list_path);
    std::istream_iterator<std::string> start(is), end;
    std::vector<std::string> testList(start, end);
    return testList;
}

 std::map<std::string, std::vector<int>>
 ReadLandmarksList(const std::string& list_path)
 {
    std::ifstream is(list_path);
    std::istream_iterator<std::string> start(is), end;
    std::vector<std::string> landmarks_list(start, end);

    std::map<std::string, std::vector<int>> landmarks_map;

    for (int i = 0; i < landmarks_list.size(); i = i + 11)
    {
        auto key = landmarks_list[i]
        
        std::vector<int> landmarks;
        for (int j = 0; j < 10; ++j)
        {
            landmarks.push_back(std::stoi(landmarks_list[i + j]));
        }

        landmarks_map[key] = landmarks;
    }
 }
