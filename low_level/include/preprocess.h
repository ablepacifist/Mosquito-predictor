#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <string>
#include <vector>
#include <ctime>

// Structure to hold the preprocessed data
struct PreprocessedData {
    std::vector<float> X_weather_cnn;             // Flattened weather data
    std::vector<std::vector<double>> X_site;      // Site data matrix
    std::vector<int> labels;                      // Breeding labels (0 or 1)
    std::vector<int> trainIndices;                // Indices for training samples
    std::vector<int> valIndices;                  // Indices for validation samples
    std::vector<int> testIndices;                 // Indices for test samples
};

// Function declarations
PreprocessedData loadAndPreprocessData(const std::string &filePath);
std::tm parseDate(const std::string &dateStr);
int dateToOrdinal(const std::tm &tm);
void standardScale(std::vector<std::vector<double>> &data);
std::vector<std::vector<int>> oneHotEncode(const std::vector<std::string> &column);
std::vector<double> fillMissingValues(const std::vector<double> &column, double defaultValue);
std::vector<std::string> fillMissingCategorical(const std::vector<std::string> &column);

#endif // PREPROCESS_H