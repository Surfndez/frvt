#ifndef TESTUTILS_H_
#define TESTUTILS_H_

#include <string>
#include <vector>
#include <map>

#include <opencv2/core.hpp>

// Read lists

std::vector<std::string> ReadTestList(const std::string& list_path);

std::map<std::string, std::vector<int>> ReadLandmarksList(const std::string& list_path);

std::vector<std::string> SplitString(const std::string& str);

// Inference

cv::Mat LoadImage(const std::string& path);

// Accuracy

double GetThreshold(double fpr_divider, std::vector<double>& diff_scores, int& borderIndex);

double CalculateTPR(double fpr_divider, std::vector<double>& diff_scores, std::vector<double>& same_scores);

#endif /* TESTUTILS_H_ */