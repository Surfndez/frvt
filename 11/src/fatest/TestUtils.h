#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>
#include <map>

std::vector<std::string> ReadTestList(const std::string& list_path);

std::map<std::string, std::vector<int>> ReadLandmarksList(const std::string& list_path);

#endif /* UTILS_H_ */
