#include "preprocess.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <ctime>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

// ----------------------------------------------------------------------------

// Function to print the first 10 data points for both branches
void printFirst10DataPoints(const std::vector<std::vector<double>> &X_weather,
                            const std::vector<std::vector<double>> &X_site) {
    std::cout << "First 10 data points for Weather Branch:" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(X_weather.size())); i++) {
        std::cout << "Data Point " << i + 1 << ": ";
        for (const auto &value : X_weather[i]) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nFirst 10 data points for Site Branch:" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(X_site.size())); i++) {
        std::cout << "Data Point " << i + 1 << ": ";
        for (const auto &value : X_site[i]) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

// ----------------------------------------------------------------------------
// Helper: split
// Splits a string by the given delimiter.
std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// ----------------------------------------------------------------------------
// Helper: trim
// Trims whitespace from the beginning and end of the string.
std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ----------------------------------------------------------------------------
// Helper: getCategoryIndex
// Searches the vector 'categories' for the given category string 's'.
// If found, returns its index. Otherwise, appends 's' and returns the new index.
// This maintains the order of first appearance.
int getCategoryIndex(std::vector<std::string> &categories, const std::string &s) {
    for (size_t i = 0; i < categories.size(); i++) {
        if (categories[i] == s)
            return static_cast<int>(i);
    }
    categories.push_back(s);
    return static_cast<int>(categories.size() - 1);
}

// ----------------------------------------------------------------------------
// Helper: parseDateOrdinal
// Expects a date string in "MM/DD/YYYY" format.
// Returns an ordinal value matching Python's date.toordinal().
// (mktime provides days since January 1, 1970; we add 719163 to match Python.)
int parseDateOrdinal(const std::string &dateStr, int &month, int &day, int &year) {
    std::tm tm = {};
    std::istringstream ss(dateStr);
    ss >> std::get_time(&tm, "%m/%d/%Y");
    if (ss.fail()) {
        month = day = year = 0;
        return 0;
    }
    month = tm.tm_mon + 1; // tm_mon is 0-indexed.
    day = tm.tm_mday;
    year = tm.tm_year + 1900;
    time_t time_temp = mktime(&tm);
    int daySince1970 = static_cast<int>(time_temp / 86400);
    int dayOrdinal = daySince1970 + 719163; // Adjust to Python's epoch.
    return dayOrdinal;
}

// ----------------------------------------------------------------------------
// Helper: computeMeanStd
// Computes the mean and standard deviation for each column in the data matrix.
void computeMeanStd(const std::vector<std::vector<double>> &data,
                    std::vector<double> &mean,
                    std::vector<double> &stdDev) {
    if (data.empty())
        return;
    int n = data.size();
    int m = data[0].size();
    mean.assign(m, 0.0);
    stdDev.assign(m, 0.0);
    for (const auto &row : data)
        for (int j = 0; j < m; j++)
            mean[j] += row[j];
    for (int j = 0; j < m; j++)
        mean[j] /= n;
    for (const auto &row : data)
        for (int j = 0; j < m; j++) {
            double diff = row[j] - mean[j];
            stdDev[j] += diff * diff;
        }
    for (int j = 0; j < m; j++) {
        stdDev[j] = std::sqrt(stdDev[j] / n);
        if (stdDev[j] == 0.0)
            stdDev[j] = 1.0; // Avoid division by zero.
    }
}

// ----------------------------------------------------------------------------
// Helper: standardizeData
// Standardizes the data (zero mean, unit variance) using the provided means and standard deviations.
void standardizeData(std::vector<std::vector<double>> &data,
                     const std::vector<double> &mean,
                     const std::vector<double> &stdDev) {
    int n = data.size();
    int m = data[0].size();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            data[i][j] = (data[i][j] - mean[j]) / stdDev[j];
}

// ----------------------------------------------------------------------------
// Helper: shuffleVector
// Shuffles a vector using a fixed seed (default: 42) to mimic Python's random_state.
template <typename T>
void shuffleVector(std::vector<T> &vec, unsigned int seed = 42) {
    std::default_random_engine engine(seed);
    std::shuffle(vec.begin(), vec.end(), engine);
}

// ----------------------------------------------------------------------------
// preprocessData
// Reads the CSV file, processes the data (including parsing dates, handling missing
// numeric values, label encoding, one-hot encoding for categorical fields, standard scaling),
// and splits the data into training (70%), validation (15%), and test (15%) sets.
// Note: This version uses order-preserving categorical encoding.
void preprocessData(const std::string &filename,
                    std::vector<std::vector<double>> &X_weather_train,
                    std::vector<std::vector<double>> &X_site_train,
                    std::vector<int> &y_train,
                    std::vector<std::vector<double>> &X_weather_val,
                    std::vector<std::vector<double>> &X_site_val,
                    std::vector<int> &y_val,
                    std::vector<std::vector<double>> &X_weather_test,
                    std::vector<std::vector<double>> &X_site_test,
                    std::vector<int> &y_test) {
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::string line;
    // Read header (we ignore its contents here).
    std::getline(file, line);
    
    // Expected header order:
    // SiteID,WetlandType,MosquitoRank,CattailMosquito,CulexFound,TreatedBy,Date,Wetness,Action,DipCount,Material,Temperature,Precipitation,CloudCoverage
    enum { IDX_SITEID = 0, IDX_WETLANDTYPE = 1, IDX_MOSQUITORANK = 2, IDX_CATTail = 3,
           IDX_CULEX = 4, IDX_TREATEDBY = 5, IDX_DATE = 6, IDX_WETNESS = 7,
           IDX_ACTION = 8, IDX_DIPCOUNT = 9, IDX_MATERIAL = 10, IDX_TEMPERATURE = 11,
           IDX_PRECIPITATION = 12, IDX_CLOUD = 13 };
    
    struct RawRow {
        std::string SiteID;
        std::string WetlandType;
        std::string MosquitoRank;
        std::string CattailMosquito;
        std::string CulexFound;
        std::string TreatedBy;
        std::string Date;
        std::string Wetness;
        std::string Action;
        std::string DipCount;
        std::string Material;
        std::string Temperature;
        std::string Precipitation;
        std::string CloudCoverage;
    };
    
    std::vector<RawRow> rawRows;
    while (std::getline(file, line)) {
        if (line.empty())
            continue;
        std::vector<std::string> tokens = split(line, ',');
        // Ensure we have 14 tokens.
        while (tokens.size() < 14)
            tokens.push_back("");
        RawRow row;
        row.SiteID           = trim(tokens[IDX_SITEID]);
        row.WetlandType      = trim(tokens[IDX_WETLANDTYPE]);
        row.MosquitoRank     = trim(tokens[IDX_MOSQUITORANK]);
        row.CattailMosquito  = trim(tokens[IDX_CATTail]);
        row.CulexFound       = trim(tokens[IDX_CULEX]);
        row.TreatedBy        = trim(tokens[IDX_TREATEDBY]);
        row.Date             = trim(tokens[IDX_DATE]);
        row.Wetness          = trim(tokens[IDX_WETNESS]);
        row.Action           = trim(tokens[IDX_ACTION]);
        row.DipCount         = trim(tokens[IDX_DIPCOUNT]);
        row.Material         = trim(tokens[IDX_MATERIAL]);
        row.Temperature      = trim(tokens[IDX_TEMPERATURE]);
        row.Precipitation    = trim(tokens[IDX_PRECIPITATION]);
        row.CloudCoverage    = trim(tokens[IDX_CLOUD]);
        rawRows.push_back(row);
    }
    file.close();
    
    // For order-preserving encoding, use vectors for each categorical field.
    std::vector<std::string> siteID_categories;
    std::vector<std::string> mosqRank_categories;
    std::vector<std::string> cattail_categories;
    std::vector<std::string> culex_categories;
    std::vector<std::string> treatedBy_categories;
    std::vector<std::string> action_categories;
    std::vector<std::string> material_categories;
    
    struct ProcessedRow {
        int breeding;        // 1 if DipCount > 0.1, else 0.
        double temperature;
        double precipitation;
        double cloud;
        int dateOrdinal;
        int month;
        int day;
        int year;
        int siteID;         // Encoded based on order of appearance.
        double wetlandType;
        double wetness;
        double dipCount;
        int mosqRank;       // Encoded category index.
        int cattail;
        int culex;
        int treatedBy;
        int action;
        int material;
    };
    
    std::vector<ProcessedRow> processed;
    for (const auto &r : rawRows) {
        ProcessedRow pr;
        double dip = 0.0;
        try {
            dip = r.DipCount.empty() ? 0.0 : std::stod(r.DipCount);
        } catch (...) {
            dip = 0.0;
        }
        pr.dipCount = dip;
        pr.breeding = (dip > 0.1) ? 1 : 0;
        
        int mon, d, yr;
        int dateOrd = 0;
        if (!r.Date.empty())
            dateOrd = parseDateOrdinal(r.Date, mon, d, yr);
        pr.dateOrdinal = dateOrd;
        pr.month = mon;
        pr.day = d;
        pr.year = yr;
        
        try { pr.temperature = r.Temperature.empty() ? 0.0 : std::stod(r.Temperature); } catch (...) { pr.temperature = 0.0; }
        try { pr.precipitation = r.Precipitation.empty() ? 0.0 : std::stod(r.Precipitation); } catch (...) { pr.precipitation = 0.0; }
        try { pr.cloud = r.CloudCoverage.empty() ? 0.0 : std::stod(r.CloudCoverage); } catch (...) { pr.cloud = 0.0; }
        
        try { pr.wetlandType = r.WetlandType.empty() ? 0.0 : std::stod(r.WetlandType); } catch (...) { pr.wetlandType = 0.0; }
        try { pr.wetness = r.Wetness.empty() ? 0.0 : std::stod(r.Wetness); } catch (...) { pr.wetness = 0.0; }
        
        // Use "Missing" if the field is empty.
        std::string sSiteID = r.SiteID.empty() ? "Missing" : r.SiteID;
        pr.siteID = getCategoryIndex(siteID_categories, sSiteID);
        
        std::string sMosqRank = r.MosquitoRank.empty() ? "Missing" : r.MosquitoRank;
        pr.mosqRank = getCategoryIndex(mosqRank_categories, sMosqRank);
        
        std::string sCattail = r.CattailMosquito.empty() ? "Missing" : r.CattailMosquito;
        pr.cattail = getCategoryIndex(cattail_categories, sCattail);
        
        std::string sCulex = r.CulexFound.empty() ? "Missing" : r.CulexFound;
        pr.culex = getCategoryIndex(culex_categories, sCulex);
        
        std::string sTreated = r.TreatedBy.empty() ? "Missing" : r.TreatedBy;
        pr.treatedBy = getCategoryIndex(treatedBy_categories, sTreated);
        
        std::string sAction = r.Action.empty() ? "Missing" : r.Action;
        pr.action = getCategoryIndex(action_categories, sAction);
        
        std::string sMaterial = r.Material.empty() ? "Missing" : r.Material;
        pr.material = getCategoryIndex(material_categories, sMaterial);
        
        processed.push_back(pr);
    }
    
    // Shuffle processed rows to mimic Python's train_test_split(random_state=42)
    std::default_random_engine engine(42);
    std::shuffle(processed.begin(), processed.end(), engine);
    
    int num_samples = processed.size();
    
    // Weather features: [Temperature, Precipitation, CloudCoverage, DateOrdinal, Month, Day, Year]
    int weather_dim = 7;
    std::vector<std::vector<double>> all_weather(num_samples, std::vector<double>(weather_dim, 0.0));
    
    // Site features: [SiteID, WetlandType, Wetness, DipCount] plus one-hot columns
    // for: MosquitoRank, CattailMosquito, CulexFound, TreatedBy, Action, Material.
    int site_dim = 4 + 
                   static_cast<int>(mosqRank_categories.size()) +
                   static_cast<int>(cattail_categories.size()) +
                   static_cast<int>(culex_categories.size()) +
                   static_cast<int>(treatedBy_categories.size()) +
                   static_cast<int>(action_categories.size()) +
                   static_cast<int>(material_categories.size());
    std::vector<std::vector<double>> all_site(num_samples, std::vector<double>(site_dim, 0.0));
    
    std::vector<int> all_labels(num_samples, 0);
    
    for (int i = 0; i < num_samples; i++) {
        const ProcessedRow &pr = processed[i];
        // Weather: Temperature, Precipitation, Cloud, DateOrdinal, Month, Day, Year.
        all_weather[i][0] = pr.temperature;
        all_weather[i][1] = pr.precipitation;
        all_weather[i][2] = pr.cloud;
        all_weather[i][3] = pr.dateOrdinal;
        all_weather[i][4] = pr.month;
        all_weather[i][5] = pr.day;
        all_weather[i][6] = pr.year;
        all_labels[i] = pr.breeding;
        
        int idx = 0;
        // Insert numeric site features: SiteID (encoded index), WetlandType, Wetness, DipCount.
        all_site[i][idx++] = pr.siteID;
        all_site[i][idx++] = pr.wetlandType;
        all_site[i][idx++] = pr.wetness;
        all_site[i][idx++] = pr.dipCount;
        
        // One-hot encoding for MosquitoRank – order preserved as in mosqRank_categories.
        for (int j = 0; j < static_cast<int>(mosqRank_categories.size()); j++) {
            all_site[i][idx++] = (j == pr.mosqRank) ? 1.0 : 0.0;
        }
        // One-hot for CattailMosquito.
        for (int j = 0; j < static_cast<int>(cattail_categories.size()); j++) {
            all_site[i][idx++] = (j == pr.cattail) ? 1.0 : 0.0;
        }
        // One-hot for CulexFound.
        for (int j = 0; j < static_cast<int>(culex_categories.size()); j++) {
            all_site[i][idx++] = (j == pr.culex) ? 1.0 : 0.0;
        }
        // One-hot for TreatedBy.
        for (int j = 0; j < static_cast<int>(treatedBy_categories.size()); j++) {
            all_site[i][idx++] = (j == pr.treatedBy) ? 1.0 : 0.0;
        }
        // One-hot for Action.
        for (int j = 0; j < static_cast<int>(action_categories.size()); j++) {
            all_site[i][idx++] = (j == pr.action) ? 1.0 : 0.0;
        }
        // One-hot for Material.
        for (int j = 0; j < static_cast<int>(material_categories.size()); j++) {
            all_site[i][idx++] = (j == pr.material) ? 1.0 : 0.0;
        }
    }
    
    // Standardize weather and site matrices separately.
    std::vector<double> weather_mean, weather_std;
    computeMeanStd(all_weather, weather_mean, weather_std);
    standardizeData(all_weather, weather_mean, weather_std);
    
    std::vector<double> site_mean, site_std;
    computeMeanStd(all_site, site_mean, site_std);
    standardizeData(all_site, site_mean, site_std);
    
    // Split into training (70%), validation (15%), and test (15%).
    int train_end = num_samples * 70 / 100;
    int val_end = num_samples * 85 / 100;
    
    X_weather_train.assign(all_weather.begin(), all_weather.begin() + train_end);
    X_site_train.assign(all_site.begin(), all_site.begin() + train_end);
    y_train.assign(all_labels.begin(), all_labels.begin() + train_end);
    
    X_weather_val.assign(all_weather.begin() + train_end, all_weather.begin() + val_end);
    X_site_val.assign(all_site.begin() + train_end, all_site.begin() + val_end);
    y_val.assign(all_labels.begin() + train_end, all_labels.begin() + val_end);
    
    X_weather_test.assign(all_weather.begin() + val_end, all_weather.end());
    X_site_test.assign(all_site.begin() + val_end, all_site.end());
    y_test.assign(all_labels.begin() + val_end, all_labels.end());

    // --- Debug Print: Show first 5 samples from training set.
    std::cout << "----- Preprocessed Data Debug (first 5 training samples) -----" << std::endl;
    int numSamplesToPrint = std::min(5, static_cast<int>(X_weather_train.size()));
    for (int i = 0; i < numSamplesToPrint; i++) {
        std::cout << "Sample " << i << " Weather: ";
        for (double val : X_weather_train[i])
            std::cout << val << " ";
        std::cout << "\nSample " << i << " Site: ";
        for (double val : X_site_train[i])
            std::cout << val << " ";
        std::cout << "\nLabel: " << y_train[i] << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
    }
}
