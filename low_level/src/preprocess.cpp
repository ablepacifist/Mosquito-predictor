#include "../include/preprocess.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <cmath>  // For std::floor and std::round
// Include a CSV parser library – make sure rapidcsv.hpp is in your include path.
#include "rapidcsv.hpp"

// --- Helper Functions ---
float interpolate(float previousValue, float nextValue) {
    return (previousValue + nextValue) / 2.0f; // Simple average of neighbors
}
// Returns the mean of valid (non-missing) float values in the vector.
float calculateMean(const std::vector<float>& vec) {
    float sum = 0.0f;
    int count = 0;
    for (const auto &v : vec) {
        if (!std::isnan(v) && !std::isinf(v)) {
            sum += v;
            count++;
        }
    }
    return (count > 0) ? sum / count : 0.0f;
}

// For NN-friendly features (e.g., Temperature),
// replace missing or invalid values (NaN or inf) with the column’s mean.
void imputeColumnWithMean(std::vector<float>& vec) {
    float mean = calculateMean(vec);
    for (auto& v : vec) {
        if (std::isnan(v) || std::isinf(v)) {
            v = mean;
        }
    }
}

// For label–like columns (e.g., DipCount), perform linear interpolation.
// This function fills consecutive missing values by interpolating between the nearest valid neighbors.
// If the first or last values are missing, they are replaced with the first/last valid value.
void interpolateColumn(std::vector<float>& vec) {
    size_t n = vec.size();
    if (n == 0) return;

    // Find index of first valid value.
    size_t firstValid = 0;
    while (firstValid < n && (std::isnan(vec[firstValid]) || std::isinf(vec[firstValid])))
        firstValid++;
    if (firstValid == n) {
        // All values are missing. Choose a default (could also flag an error).
        for (size_t i = 0; i < n; i++) {
            vec[i] = 0.0f;
        }
        return;
    }
    // Replace any missing values at the beginning with the first valid value.
    for (size_t i = 0; i < firstValid; i++) {
        vec[i] = vec[firstValid];
    }
    // Process the rest of the column.
    size_t i = firstValid;
    while (i < n) {
        if (!std::isnan(vec[i]) && !std::isinf(vec[i])) {
            i++;
            continue;
        }
        // Find previous valid value (guaranteed to be valid since i > 0)
        size_t start = i - 1;
        // Find next valid index.
        size_t j = i;
        while (j < n && (std::isnan(vec[j]) || std::isinf(vec[j])))
            j++;
        if (j < n) {
            float startVal = vec[start];
            float endVal = vec[j];
            size_t gap = j - start;
            for (size_t k = i; k < j; k++) {
                // Linear interpolation: proportionally fill between startVal and endVal.
                vec[k] = startVal + (endVal - startVal) * (float)(k - start) / (float)gap;
            }
            i = j;
        } else {
            // No valid value after; fill all remaining with the last valid value.
            for (size_t k = i; k < n; k++) {
                vec[k] = vec[start];
            }
            break;
        }
    }
}

//parse a numeric column (if used in site features).
std::vector<float> parseNumericColumn(rapidcsv::Document &doc, const std::string &columnName) {
    std::vector<float> parsedColumn;
    try {
        std::vector<std::string> rawColumn = doc.GetColumn<std::string>(columnName);
        for (const auto &value : rawColumn) {
            try {
                parsedColumn.push_back(std::stof(value));
            } catch (...) {
                parsedColumn.push_back(NAN); // Mark invalid/missing as NaN
            }
        }
        // Impute missing values with the column's mean (for NN-friendly features)
        imputeColumnWithMean(parsedColumn);
    } catch (const std::exception &e) {
        throw std::runtime_error("Error parsing numeric column '" + columnName + "': " + std::string(e.what()));
    }
    return parsedColumn;
}
/**
 * @brief Parse a date in the format "mm/dd/yyyy" into a std::tm struct.
 * @param dateStr The date string to parse.
 * @return A std::tm struct representing the parsed date.
 */
std::tm parseDate(const std::string &dateStr)
{
    std::tm tm = {};
    std::istringstream ss(dateStr);
    // std::get_time is available in C++11 and later though support may vary on Windows.
    ss >> std::get_time(&tm, "%m/%d/%Y");
    if (ss.fail())
    {
        throw std::runtime_error("Failed to parse date: " + dateStr);
    }
    return tm;
}

/**
 * @brief Convert a std::tm structure to an ordinal day (e.g., days since Unix epoch).
 * @param tm The std::tm structure to convert.
 * @return The ordinal day as an integer.
 */
int dateToOrdinal(const std::tm &tm)
{
    std::time_t timeEpoch = std::mktime(const_cast<std::tm *>(&tm));
    return static_cast<int>(timeEpoch / (24 * 3600));
}

/**
 * @brief Standard scaling: subtract the mean and divide by standard deviation, column-wise.
 * @param data The 2D vector of data to scale.
 */
void standardScale(std::vector<std::vector<double>> &data)
{
    if (data.empty())
        return;
    int rows = data.size();
    int cols = data[0].size();
    std::vector<double> means(cols, 0.0);
    std::vector<double> stds(cols, 0.0);

    // Compute means
    for (int j = 0; j < cols; j++)
    {
        double sum = 0;
        for (int i = 0; i < rows; i++)
        {
            sum += data[i][j];
        }
        means[j] = sum / rows;
    }

    // Compute standard deviations
    for (int j = 0; j < cols; j++)
    {
        double sumSq = 0;
        for (int i = 0; i < rows; i++)
        {
            double diff = data[i][j] - means[j];
            sumSq += diff * diff;
        }
        stds[j] = std::sqrt(sumSq / rows);
        if (stds[j] == 0)
            stds[j] = 1; // avoid dividing by zero
    }

    // Scale data
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            data[i][j] = (data[i][j] - means[j]) / stds[j];
        }
    }
}

/**
 * @brief One-hot encode a categorical column.
 * @param column The vector of categorical values to encode.
 * @return A 2D vector representing the one-hot encoded data.
 */
std::vector<std::vector<int>> oneHotEncode(const std::vector<std::string> &column)
{
    std::unordered_map<std::string, int> categoryMap;
    int code = 0;
    for (const auto &value : column)
    {
        if (categoryMap.find(value) == categoryMap.end())
        {
            categoryMap[value] = code++;
        }
    }

    std::vector<std::vector<int>> oneHot(column.size(), std::vector<int>(categoryMap.size(), 0));
    for (size_t i = 0; i < column.size(); i++)
    {
        oneHot[i][categoryMap[column[i]]] = 1;
    }
    return oneHot;
}

/**
 * @brief Fill missing values in a numeric column with a default value.
 * @param column The vector of numeric values.
 * @param defaultValue The default value to use for missing values.
 * @return A vector with missing values replaced.
 */
std::vector<double> fillMissingValues(const std::vector<double> &column, double defaultValue)
{
    std::vector<double> filledColumn = column;
    for (auto &value : filledColumn)
    {
        if (std::isnan(value))
        {
            value = defaultValue;
        }
    }
    return filledColumn;
}

/**
 * @brief Replace missing values in a categorical column with "missing".
 * @param column The vector of categorical values.
 * @return A vector with missing values replaced.
 */
std::vector<std::string> fillMissingCategorical(const std::vector<std::string> &column)
{
    std::vector<std::string> filledColumn = column;
    for (auto &value : filledColumn)
    {
        if (value.empty())
        {
            value = "missing";
        }
    }
    return filledColumn;
}

// --- Preprocess Class Implementation ---

/**
 * @brief Preprocess the data from a CSV file.
 * @param filePath The path to the CSV file.
 * @param X_weather_train Output: Weather training data.
 * @param X_site_train Output: Site training data.
 * @param y_train Output: Training labels.
 * @param X_weather_val Output: Weather validation data.
 * @param X_site_val Output: Site validation data.
 * @param y_val Output: Validation labels.
 * @param X_weather_test Output: Weather test data.
 * @param X_site_test Output: Site test data.
 * @param y_test Output: Test labels.
 */
void preprocessData(const std::string &filePath,
                    std::vector<std::vector<double>> &X_weather_train, std::vector<std::vector<double>> &X_site_train, std::vector<int> &y_train,
                    std::vector<std::vector<double>> &X_weather_val, std::vector<std::vector<double>> &X_site_val, std::vector<int> &y_val,
                    std::vector<std::vector<double>> &X_weather_test, std::vector<std::vector<double>> &X_site_test, std::vector<int> &y_test)
{
    try
    {
        // Load the CSV file
        rapidcsv::Document doc(filePath);

        // Parse labels
        auto y = parseLabels(doc);

        // Parse weather features
        auto X_weather = parseWeatherFeatures(doc);

        // Parse site features
        auto X_site = parseSiteFeatures(doc);

        // Normalize numeric data
        standardScale(X_weather);
        standardScale(X_site);

        // Split data into training, validation, and test sets
        splitData(X_weather, X_site, y, X_weather_train, X_site_train, y_train, X_weather_val, X_site_val, y_val, X_weather_test, X_site_test, y_test);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in preprocessData: " << e.what() << std::endl;
        throw; // Re-throw the exception for debugging purposes
    }
}

/**
 * @brief Parse labels from the "DipCount" column in the CSV file.
 * @param doc The rapidcsv::Document object representing the CSV file.
 * @return A vector of labels (0 or 1).
 */
std::vector<int> parseLabels(rapidcsv::Document &doc) {
    std::vector<float> dipCount;
    try {
        std::vector<std::string> dipCountRaw = doc.GetColumn<std::string>("DipCount");
        for (const auto &value : dipCountRaw) {
            try {
                dipCount.push_back(std::stof(value));
            } catch (...) {
                dipCount.push_back(NAN);
            }
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Error parsing 'DipCount' column: " + std::string(e.what()));
    }

    // Interpolate missing values in DipCount to preserve numerical continuity.
    interpolateColumn(dipCount);

    // For label creation, e.g., thresholding at 0.1
    std::vector<int> y;
    for (const auto &value : dipCount) {
        y.push_back(value > 0.1 ? 1 : 0);
    }
    return y;
}

/**
 * @brief Parse weather-related features from the CSV file.
 * @param doc The rapidcsv::Document object representing the CSV file.
 * @return A 2D vector of weather features.
 */
std::vector<std::vector<double>> parseWeatherFeatures(rapidcsv::Document &doc)
{
    std::vector<float> temperature, precipitation, cloudCoverage, dateOrdinal;
    try
    {
        // Get the numeric columns as strings so we can convert them manually.
        std::vector<std::string> tempStr = doc.GetColumn<std::string>("Temperature");
        std::vector<std::string> precipStr = doc.GetColumn<std::string>("Precipitation");
        std::vector<std::string> cloudStr = doc.GetColumn<std::string>("CloudCoverage");

        // Resize our float vectors to match the number of rows.
        temperature.resize(tempStr.size());
        precipitation.resize(precipStr.size());
        cloudCoverage.resize(cloudStr.size());

        // Convert temperature strings to floats manually.
        for (size_t i = 0; i < tempStr.size(); i++)
        {
            try {
                temperature[i] = std::stof(tempStr[i]);
            } catch (...) {
                temperature[i] = NAN;
            }
        }
        // Convert precipitation strings to floats manually.
        for (size_t i = 0; i < precipStr.size(); i++)
        {
            try {
                precipitation[i] = std::stof(precipStr[i]);
            } catch (...) {
                precipitation[i] = NAN;
            }
        }
        // Convert cloud coverage strings to floats manually.
        for (size_t i = 0; i < cloudStr.size(); i++)
        {
            try {
                cloudCoverage[i] = std::stof(cloudStr[i]);
            } catch (...) {
                cloudCoverage[i] = NAN;
            }
        }

        // Impute missing or invalid values using the column mean.
        imputeColumnWithMean(temperature);
        imputeColumnWithMean(precipitation);
        imputeColumnWithMean(cloudCoverage);

        // Parse the dates and convert them to ordinal values.
        std::vector<std::string> dates = doc.GetColumn<std::string>("Date");
        for (const auto &date : dates)
        {
            try
            {
                std::tm tm = parseDate(date); // Make sure your parseDate handles errors appropriately.
                dateOrdinal.push_back(static_cast<float>(dateToOrdinal(tm)));
            }
            catch (...)
            {
                dateOrdinal.push_back(0.0f);
            }
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Error parsing weather features: " + std::string(e.what()));
    }

    // Build and return the weather features matrix.
    std::vector<std::vector<double>> X_weather;
    for (size_t i = 0; i < temperature.size(); i++)
    {
        X_weather.push_back({ static_cast<double>(temperature[i]),
                              static_cast<double>(precipitation[i]),
                              static_cast<double>(cloudCoverage[i]),
                              static_cast<double>(dateOrdinal[i])});
    }
    return X_weather;
}

/**
 * @brief Parse site-related features from the CSV file.
 * @param doc The rapidcsv::Document object representing the CSV file.
 * @return A 2D vector of site features.
 */
std::vector<std::vector<double>> parseSiteFeatures(rapidcsv::Document &doc) {
    std::vector<std::string> siteID, mosquitoRank, cattailMosquito, treatedBy, action, material;
    std::vector<float> wetlandType, wetness;

    try {
        // Parse categorical columns.
        siteID = doc.GetColumn<std::string>("SiteID");
        mosquitoRank = getColumnWithDefault(doc, "MosquitoRank", "Missing");
        cattailMosquito = getColumnWithDefault(doc, "CattailMosquito", "No");
        treatedBy = doc.GetColumn<std::string>("TreatedBy");
        action = doc.GetColumn<std::string>("Action");
        material = doc.GetColumn<std::string>("Material");

        // Parse numeric columns using our helper.
        wetlandType = parseNumericColumn(doc, "WetlandType");
        wetness = parseNumericColumn(doc, "Wetness");
    } catch (const std::exception &e) {
        throw std::runtime_error("Error parsing site features: " + std::string(e.what()));
    }

    // Assume oneHotEncode is defined elsewhere.
    auto siteEncoded = oneHotEncode(siteID);
    auto mosquitoRankEncoded = oneHotEncode(mosquitoRank);
    auto cattailMosquitoEncoded = oneHotEncode(cattailMosquito);
    auto treatedByEncoded = oneHotEncode(treatedBy);
    auto actionEncoded = oneHotEncode(action);
    auto materialEncoded = oneHotEncode(material);

    std::vector<std::vector<double>> X_site;
    for (size_t i = 0; i < wetlandType.size(); i++) {
        std::vector<double> siteRow = { static_cast<double>(wetlandType[i]), static_cast<double>(wetness[i]) };
        for (const auto &encoded : { siteEncoded[i], mosquitoRankEncoded[i], 
                                      cattailMosquitoEncoded[i], treatedByEncoded[i],
                                      actionEncoded[i], materialEncoded[i] })
        {
            siteRow.insert(siteRow.end(), encoded.begin(), encoded.end());
        }
        X_site.push_back(siteRow);
    }
    return X_site;
}

/**
 * @brief Get a column from the CSV file, or return a default value if the column is missing.
 * @param doc The rapidcsv::Document object representing the CSV file.
 * @param columnName The name of the column to retrieve.
 * @param defaultValue The default value to use if the column is missing.
 * @return A vector of strings representing the column values.
 */
std::vector<std::string> getColumnWithDefault(rapidcsv::Document &doc, const std::string &columnName, const std::string &defaultValue)
{
    try
    {
        return doc.GetColumn<std::string>(columnName);
    }
    catch (...)
    {
        return std::vector<std::string>(doc.GetRowCount(), defaultValue);
    }
}

/**
 * @brief Split data into training, validation, and test sets.
 * @param X_weather The weather features.
 * @param X_site The site features.
 * @param y The labels.
 * @param X_weather_train Output: Weather training data.
 * @param X_site_train Output: Site training data.
 * @param y_train Output: Training labels.
 * @param X_weather_val Output: Weather validation data.
 * @param X_site_val Output: Site validation data.
 * @param y_val Output: Validation labels.
 * @param X_weather_test Output: Weather test data.
 * @param X_site_test Output: Site test data.
 * @param y_test Output: Test labels.
 */

void splitData(const std::vector<std::vector<double>> &X_weather, 
               const std::vector<std::vector<double>> &X_site, 
               const std::vector<int> &y,
               std::vector<std::vector<double>> &X_weather_train, 
               std::vector<std::vector<double>> &X_site_train, 
               std::vector<int> &y_train,
               std::vector<std::vector<double>> &X_weather_val, 
               std::vector<std::vector<double>> &X_site_val, 
               std::vector<int> &y_val,
               std::vector<std::vector<double>> &X_weather_test, 
               std::vector<std::vector<double>> &X_site_test, 
               std::vector<int> &y_test)
{
    size_t dataSize = y.size();
    if (dataSize == 0) return;

    // Calculate sizes:
    // Training gets floor(0.7 * n)
    size_t trainSize = static_cast<size_t>(std::floor(dataSize * 0.7));
    
    // The remaining rows will be split equally between validation and test.
    size_t remaining = dataSize - trainSize;
    size_t valSize = static_cast<size_t>(std::round(remaining * 0.5)); // since 15%/30% = 0.5
    size_t testSize = remaining - valSize;  // The rest goes to testing

    // Use these counts to assign rows sequentially.
    for (size_t i = 0; i < dataSize; i++) {
        if (i < trainSize) {
            X_weather_train.push_back(X_weather[i]);
            X_site_train.push_back(X_site[i]);
            y_train.push_back(y[i]);
        }
        else if (i < trainSize + valSize) {
            X_weather_val.push_back(X_weather[i]);
            X_site_val.push_back(X_site[i]);
            y_val.push_back(y[i]);
        }
        else {
            X_weather_test.push_back(X_weather[i]);
            X_site_test.push_back(X_site[i]);
            y_test.push_back(y[i]);
        }
    }
}

/**
 * @brief Parse the raw DipCount column as continuous values for regression.
 * @param doc The rapidcsv::Document object representing the CSV file.
 * @return A vector of float values representing the raw DipCount (with missing values interpolated).
 */
std::vector<float> parseDipCountReg(rapidcsv::Document &doc) {
    std::vector<float> dipCount;
    try {
        std::vector<std::string> dipCountRaw = doc.GetColumn<std::string>("DipCount");
        for (const auto &value : dipCountRaw) {
            try {
                dipCount.push_back(std::stof(value));
            } catch (...) {
                dipCount.push_back(NAN); // Mark missing/invalid values as NaN.
            }
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Error parsing 'DipCount' column for regression: " + std::string(e.what()));
    }

    // Interpolate missing values to preserve continuity.
    interpolateColumn(dipCount);
    return dipCount;
}

/**
 * @brief Split data into training, validation, and test sets, with labels as floats for regression.
 * @param X_weather The weather features.
 * @param X_site The site features.
 * @param y A vector of continuous labels.
 * @param X_weather_train Output: Weather training data.
 * @param X_site_train Output: Site training data.
 * @param y_train Output: Training labels (floats).
 * @param X_weather_val Output: Weather validation data.
 * @param X_site_val Output: Site validation data.
 * @param y_val Output: Validation labels (floats).
 * @param X_weather_test Output: Weather test data.
 * @param X_site_test Output: Site test data.
 * @param y_test Output: Test labels (floats).
 */
void splitDataRegression(const std::vector<std::vector<double>> &X_weather, 
                         const std::vector<std::vector<double>> &X_site, 
                         const std::vector<float> &y,
                         std::vector<std::vector<double>> &X_weather_train, 
                         std::vector<std::vector<double>> &X_site_train, 
                         std::vector<float> &y_train,
                         std::vector<std::vector<double>> &X_weather_val, 
                         std::vector<std::vector<double>> &X_site_val, 
                         std::vector<float> &y_val,
                         std::vector<std::vector<double>> &X_weather_test, 
                         std::vector<std::vector<double>> &X_site_test, 
                         std::vector<float> &y_test)
{
    size_t dataSize = y.size();
    if (dataSize == 0) return;

    // Use 70% for training.
    size_t trainSize = static_cast<size_t>(std::floor(dataSize * 0.7));
    // Split the rest equally between validation and test.
    size_t remaining = dataSize - trainSize;
    size_t valSize = static_cast<size_t>(std::round(remaining * 0.5));
    size_t testSize = remaining - valSize;

    for (size_t i = 0; i < dataSize; i++) {
        if (i < trainSize) {
            X_weather_train.push_back(X_weather[i]);
            X_site_train.push_back(X_site[i]);
            y_train.push_back(y[i]);
        } else if (i < trainSize + valSize) {
            X_weather_val.push_back(X_weather[i]);
            X_site_val.push_back(X_site[i]);
            y_val.push_back(y[i]);
        } else {
            X_weather_test.push_back(X_weather[i]);
            X_site_test.push_back(X_site[i]);
            y_test.push_back(y[i]);
        }
    }
}

/**
 * @brief Preprocess the data from a CSV file for regression.
 *
 * This function reads the CSV file, parses weather features, site features, and continuous dip count labels,
 * performs standard scaling, and splits the data into training, validation, and test sets.
 *
 * @param filePath The path to the CSV file.
 * @param X_weather_train Output: Weather training data.
 * @param X_site_train Output: Site training data.
 * @param y_train Output: Training labels (as floats).
 * @param X_weather_val Output: Weather validation data.
 * @param X_site_val Output: Site validation data.
 * @param y_val Output: Validation labels (as floats).
 * @param X_weather_test Output: Weather test data.
 * @param X_site_test Output: Site test data.
 * @param y_test Output: Test labels (as floats).
 */
void preprocessDataRegression(const std::string &filePath,
                              std::vector<std::vector<double>> &X_weather_train, 
                              std::vector<std::vector<double>> &X_site_train, 
                              std::vector<float> &y_train,
                              std::vector<std::vector<double>> &X_weather_val, 
                              std::vector<std::vector<double>> &X_site_val, 
                              std::vector<float> &y_val,
                              std::vector<std::vector<double>> &X_weather_test, 
                              std::vector<std::vector<double>> &X_site_test, 
                              std::vector<float> &y_test)
{
    try {
        // Load the CSV file.
        rapidcsv::Document doc(filePath);

        // Parse continuous DipCount for regression.
        auto y = parseDipCountReg(doc);

        // Parse weather and site features.
        auto X_weather = parseWeatherFeatures(doc);
        auto X_site = parseSiteFeatures(doc);

        // Normalize numeric features.
        standardScale(X_weather);
        standardScale(X_site);

        // Split the data into training, validation, and test sets.
        splitDataRegression(X_weather, X_site, y, 
                            X_weather_train, X_site_train, y_train,
                            X_weather_val, X_site_val, y_val,
                            X_weather_test, X_site_test, y_test);
    } catch (const std::exception &e) {
        std::cerr << "Error in preprocessDataRegression: " << e.what() << std::endl;
        throw; // Re-throw for debugging purposes.
    }
}

