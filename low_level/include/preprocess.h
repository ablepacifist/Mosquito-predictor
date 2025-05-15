#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <string>
#include <vector>

// This function reads a CSV file and splits the data into training, validation, and test sets.
// It produces:
// - X_weather_train: weather features for training samples.
// - X_site_train: site features for training samples.
// - y_train: labels for training samples.
// - X_weather_val: weather features for validation samples.
// - X_site_val: site features for validation samples.
// - y_val: labels for validation samples.
// - X_weather_test: weather features for test samples.
// - X_site_test: site features for test samples.
// - y_test: labels for test samples.
void preprocessData(const std::string& filename,
                    std::vector<std::vector<double>>& X_weather_train,
                    std::vector<std::vector<double>>& X_site_train,
                    std::vector<int>& y_train,
                    std::vector<std::vector<double>>& X_weather_val,
                    std::vector<std::vector<double>>& X_site_val,
                    std::vector<int>& y_val,
                    std::vector<std::vector<double>>& X_weather_test,
                    std::vector<std::vector<double>>& X_site_test,
                    std::vector<int>& y_test);

                    void printFirst10DataPoints(const std::vector<std::vector<double>> &X_weather,
        const std::vector<std::vector<double>> &X_site);
#endif // PREPROCESS_H
