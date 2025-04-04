#include "preprocess.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <set>

// Include a CSV parser library – make sure rapidcsv.hpp is in your include path.
#include "rapidcsv.hpp"

// --- Helper Functions ---

// Parse a date in the format "mm/dd/yyyy" into a std::tm struct.
std::tm parseDate(const std::string &dateStr) {
    std::tm tm = {};
    std::istringstream ss(dateStr);
    // std::get_time is available in C++11 and later though support may vary on Windows.
    ss >> std::get_time(&tm, "%m/%d/%Y");
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse date: " + dateStr);
    }
    return tm;
}

// Convert a std::tm structure to an ordinal day (e.g., days since Unix epoch).
int dateToOrdinal(const std::tm &tm) {
    std::time_t timeEpoch = std::mktime(const_cast<std::tm*>(&tm));
    return static_cast<int>(timeEpoch / (24 * 3600));
}

// Standard scaling: subtract the mean and divide by standard deviation, column-wise.
void standardScale(std::vector<std::vector<double>> &data) {
    if(data.empty()) return;
    int rows = data.size();
    int cols = data[0].size();
    std::vector<double> means(cols, 0.0);
    std::vector<double> stds(cols, 0.0);
    
    // Compute means
    for (int j = 0; j < cols; j++) {
        double sum = 0;
        for (int i = 0; i < rows; i++){
            sum += data[i][j];
        }
        means[j] = sum / rows;
    }
    
    // Compute standard deviations
    for (int j = 0; j < cols; j++){
        double sumSq = 0;
        for (int i = 0; i < rows; i++){
            double diff = data[i][j] - means[j];
            sumSq += diff * diff;
        }
        stds[j] = std::sqrt(sumSq / rows);
        if(stds[j] == 0) stds[j] = 1; // avoid dividing by zero
    }
    
    // Scale data
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            data[i][j] = (data[i][j] - means[j]) / stds[j];
        }
    }
}

// One-hot encode a categorical column
std::vector<std::vector<int>> oneHotEncode(const std::vector<std::string> &column) {
    std::unordered_map<std::string, int> categoryMap;
    int code = 0;
    for (const auto &value : column) {
        if (categoryMap.find(value) == categoryMap.end()) {
            categoryMap[value] = code++;
        }
    }

    std::vector<std::vector<int>> oneHot(column.size(), std::vector<int>(categoryMap.size(), 0));
    for (size_t i = 0; i < column.size(); i++) {
        oneHot[i][categoryMap[column[i]]] = 1;
    }
    return oneHot;
}

// Fill missing values in a numeric column with a default value
std::vector<double> fillMissingValues(const std::vector<double> &column, double defaultValue) {
    std::vector<double> filledColumn = column;
    for (auto &value : filledColumn) {
        if (std::isnan(value)) {
            value = defaultValue;
        }
    }
    return filledColumn;
}

// Replace missing values in a categorical column with "missing"
std::vector<std::string> fillMissingCategorical(const std::vector<std::string> &column) {
    std::vector<std::string> filledColumn = column;
    for (auto &value : filledColumn) {
        if (value.empty()) {
            value = "missing";
        }
    }
    return filledColumn;
}
