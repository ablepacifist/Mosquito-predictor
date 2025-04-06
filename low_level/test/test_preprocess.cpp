// File: tests/preprocessDataTests.cpp

#include <../test/gtest/gtest.h>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include "../include/preprocess.h"  // This header declares preprocessData, standardScale, parseDate, dateToOrdinal, oneHotEncode, etc.

// Mock data for testing
const std::string testFilePath = "data/combined_data.csv"; // Path to the test CSV file
// Test 1: Check that data is split according to the specified proportions
TEST(PreprocessDataTests, DataSplitProportions) {
    std::vector<std::vector<double>> X_weather_train, X_site_train;
    std::vector<std::vector<double>> X_weather_val, X_site_val;
    std::vector<std::vector<double>> X_weather_test, X_site_test;
    std::vector<int> y_train, y_val, y_test;

    preprocessData(testFilePath, 
                   X_weather_train, X_site_train, y_train, 
                   X_weather_val, X_site_val, y_val, 
                   X_weather_test, X_site_test, y_test);

    size_t totalSize = y_train.size() + y_val.size() + y_test.size();

    // Compute expected sizes with rounding logic for val/test splits
    size_t expectedTrainSize = static_cast<size_t>(std::floor(totalSize * 0.7));  // Floor for training
    size_t remaining = totalSize - expectedTrainSize;                            // Remaining rows
    size_t expectedValSize = static_cast<size_t>(std::round(remaining * 0.5));   // Rounded for validation
    size_t expectedTestSize = remaining - expectedValSize;                       // Remaining to testing

    // Verify split sizes directly
    EXPECT_EQ(y_train.size(), expectedTrainSize);
    EXPECT_EQ(y_val.size(), expectedValSize);
    EXPECT_EQ(y_test.size(), expectedTestSize);

    // Verify proportions for additional reassurance
    double trainRatio = static_cast<double>(y_train.size()) / totalSize;
    double valRatio = static_cast<double>(y_val.size()) / totalSize;
    double testRatio = static_cast<double>(y_test.size()) / totalSize;

    EXPECT_NEAR(trainRatio, 0.7, 0.01); // Tight tolerance for training
    EXPECT_NEAR(valRatio, 0.15, 0.01);  // Tight tolerance for validation
    EXPECT_NEAR(testRatio, 0.15, 0.01); // Tight tolerance for testing
}



// Test 2: Check handling of missing values in DipCount
TEST(PreprocessDataTests, DipCountMissingValues) {
    rapidcsv::Document doc(testFilePath);
    auto labels = parseLabels(doc);

    // Ensure that labels are correctly parsed and missing values are handled
    for (const auto &label : labels) {
        EXPECT_TRUE(label == 0 || label == 1);
    }
}

// Test 3: Verify one-hot encoding produces the expected encoded vector size
TEST(PreprocessDataTests, OneHotEncodingSize) {
    std::vector<std::string> categories = {"A", "B", "A", "C"};
    auto oneHotEncoded = oneHotEncode(categories);

    // Check the size of the one-hot encoded matrix
    EXPECT_EQ(oneHotEncoded.size(), categories.size());
    EXPECT_EQ(oneHotEncoded[0].size(), 3); // 3 unique categories: A, B, C
}

// Test 4: Verify standard scaling normalizes data correctly
TEST(PreprocessDataTests, StandardScaling) {
    std::vector<std::vector<double>> data = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    standardScale(data);

    // Check that each column has mean 0 and standard deviation 1
    for (size_t j = 0; j < data[0].size(); j++) {
        double mean = 0.0, stdDev = 0.0;
        for (size_t i = 0; i < data.size(); i++) {
            mean += data[i][j];
        }
        mean /= data.size();

        for (size_t i = 0; i < data.size(); i++) {
            stdDev += std::pow(data[i][j] - mean, 2);
        }
        stdDev = std::sqrt(stdDev / data.size());

        EXPECT_NEAR(mean, 0.0, 1e-6);
        EXPECT_NEAR(stdDev, 1.0, 1e-6);
    }
}

// Test 5: Verify date parsing and conversion to ordinal
TEST(PreprocessDataTests, DateParsingAndOrdinalConversion) {
    std::string dateStr = "09/08/2024";
    std::tm parsedDate = parseDate(dateStr);
    int ordinal = dateToOrdinal(parsedDate);

    // Check that the parsed date matches the expected values
    EXPECT_EQ(parsedDate.tm_year + 1900, 2024);
    EXPECT_EQ(parsedDate.tm_mon + 1, 9);
    EXPECT_EQ(parsedDate.tm_mday, 8);

    // Check that the ordinal value is consistent
    EXPECT_GT(ordinal, 0);
}

// Test 6: Verify weather features parsing
TEST(PreprocessDataTests, ParseWeatherFeatures) {
    rapidcsv::Document doc(testFilePath);
    auto weatherFeatures = parseWeatherFeatures(doc);

    // Ensure the weather features matrix has the correct dimensions
    EXPECT_GT(weatherFeatures.size(), 0);
    EXPECT_EQ(weatherFeatures[0].size(), 4); // Temperature, Precipitation, CloudCoverage, DateOrdinal
}

// Test 7: Verify site features parsing
TEST(PreprocessDataTests, ParseSiteFeatures) {
    rapidcsv::Document doc(testFilePath);
    auto siteFeatures = parseSiteFeatures(doc);

    // Ensure the site features matrix has the correct dimensions
    EXPECT_GT(siteFeatures.size(), 0);
    EXPECT_GT(siteFeatures[0].size(), 0); // Site features should have at least one column
}

// Test 8: Verify splitData function splits data correctly
TEST(PreprocessDataTests, SplitData) {
    std::vector<std::vector<double>> X_weather = {
        {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}, {9.0, 10.0}
    };
    std::vector<std::vector<double>> X_site = {
        {11.0, 12.0}, {13.0, 14.0}, {15.0, 16.0}, {17.0, 18.0}, {19.0, 20.0}
    };
    std::vector<int> y = {0, 1, 0, 1, 0};

    std::vector<std::vector<double>> X_weather_train, X_site_train;
    std::vector<std::vector<double>> X_weather_val, X_site_val;
    std::vector<std::vector<double>> X_weather_test, X_site_test;
    std::vector<int> y_train, y_val, y_test;

    splitData(X_weather, X_site, y, 
              X_weather_train, X_site_train, y_train, 
              X_weather_val, X_site_val, y_val, 
              X_weather_test, X_site_test, y_test);

    // Check the sizes of the splits
    EXPECT_EQ(X_weather_train.size(), 3); // 70% of 5
    EXPECT_EQ(X_weather_val.size(), 1);   // 15% of 5
    EXPECT_EQ(X_weather_test.size(), 1);  // Remaining 15% of 5
    EXPECT_EQ(y_train.size(), 3);
    EXPECT_EQ(y_val.size(), 1);
    EXPECT_EQ(y_test.size(), 1);
}
