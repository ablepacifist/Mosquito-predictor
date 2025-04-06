#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <string>
#include <vector>
#include <ctime>
#include "rapidcsv.hpp"

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
std::tm parseDate(const std::string &dateStr);
int dateToOrdinal(const std::tm &tm);
void standardScale(std::vector<std::vector<double>> &data);
std::vector<std::vector<int>> oneHotEncode(const std::vector<std::string> &column);
std::vector<double> fillMissingValues(const std::vector<double> &column, double defaultValue);
std::vector<std::string> fillMissingCategorical(const std::vector<std::string> &column);

std::vector<int> parseLabels(rapidcsv::Document &doc);
std::vector<std::vector<double>> parseWeatherFeatures(rapidcsv::Document &doc);
std::vector<std::vector<double>> parseSiteFeatures(rapidcsv::Document &doc);

void splitData(const std::vector<std::vector<double>> &X_weather, const std::vector<std::vector<double>> &X_site, const std::vector<int> &y,
               std::vector<std::vector<double>> &X_weather_train, std::vector<std::vector<double>> &X_site_train, std::vector<int> &y_train,
               std::vector<std::vector<double>> &X_weather_val, std::vector<std::vector<double>> &X_site_val, std::vector<int> &y_val,
               std::vector<std::vector<double>> &X_weather_test, std::vector<std::vector<double>> &X_site_test, std::vector<int> &y_test);

void preprocessData(const std::string &filePath,
    std::vector<std::vector<double>> &X_weather_train, std::vector<std::vector<double>> &X_site_train, std::vector<int> &y_train,
    std::vector<std::vector<double>> &X_weather_val, std::vector<std::vector<double>> &X_site_val, std::vector<int> &y_val,
    std::vector<std::vector<double>> &X_weather_test, std::vector<std::vector<double>> &X_site_test, std::vector<int> &y_test);

std::vector<std::string> getColumnWithDefault(rapidcsv::Document &doc, const std::string &columnName, const std::string &defaultValue);

#endif // PREPROCESS_H