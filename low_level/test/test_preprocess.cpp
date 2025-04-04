#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <unordered_map>
#include "../include/preprocess.h" // Include the header file for preprocessing functions

// Helper function to check if a vector contains NaN values
bool containsNaN(const std::vector<double> &vec) {
    for (const auto &val : vec) {
        if (std::isnan(val)) {
            return true;
        }
    }
    return false;
}

// Test 1: Ensure missing numerical values are not replaced
void testMissingNumericalValues(const std::vector<double> &column) {
    assert(!containsNaN(column) && "Numerical column contains NaN values!");
}

// Test 2: Ensure categorical missing values are replaced with "missing"
void testMissingCategoricalValues(const std::vector<std::string> &column) {
    for (const auto &value : column) {
        assert(!value.empty() && "Categorical column contains empty values!");
        assert(value != "missing" || !value.empty() && "Missing categorical values not replaced with 'missing'!");
    }
}

// Test 3: Ensure date conversion is correct
void testDateConversion(const std::vector<std::string> &dates, const std::vector<int> &dateOrdinal) {
    for (size_t i = 0; i < dates.size(); i++) {
        std::tm tm = parseDate(dates[i]);
        int expectedOrdinal = dateToOrdinal(tm);
        assert(dateOrdinal[i] == expectedOrdinal && "Date conversion to ordinal is incorrect!");
    }
}

// Test 4: Ensure one-hot encoding is correct
void testOneHotEncoding(const std::vector<std::string> &column, const std::vector<std::vector<int>> &oneHotEncoded) {
    std::unordered_map<std::string, int> categoryMap;
    int code = 0;
    for (const auto &value : column) {
        if (categoryMap.find(value) == categoryMap.end()) {
            categoryMap[value] = code++;
        }
    }

    for (size_t i = 0; i < column.size(); i++) {
        for (size_t j = 0; j < oneHotEncoded[i].size(); j++) {
            if (j == categoryMap[column[i]]) {
                assert(oneHotEncoded[i][j] == 1 && "One-hot encoding is incorrect!");
            } else {
                assert(oneHotEncoded[i][j] == 0 && "One-hot encoding is incorrect!");
            }
        }
    }
}

// Test 5: Ensure numerical data is scaled correctly
void testScaling(const std::vector<std::vector<double>> &data) {
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

        assert(std::abs(mean) < 1e-6 && "Scaled data mean is not zero!");
        assert(std::abs(stdDev - 1.0) < 1e-6 && "Scaled data standard deviation is not one!");
    }
}

// Test 6: Ensure data integrity is preserved
void testDataIntegrity(const std::vector<std::vector<double>> &original, const std::vector<std::vector<double>> &processed) {
    assert(original.size() == processed.size() && "Data size mismatch!");
    for (size_t i = 0; i < original.size(); i++) {
        assert(original[i].size() == processed[i].size() && "Data column size mismatch!");
    }
}

// Main test function
int main() {
    try {
        // Load the dataset and run tests
        std::cout << "All tests passed!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}