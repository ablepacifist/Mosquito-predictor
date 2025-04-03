# Mosquito Breeding Site Classification

This project applies a deep learning model to classify mosquito breeding sites based on data from [MMCD.org](http://mmcd.org). The current system integrates various data sources—including raw site records, weather data fetched from an external API, and preprocessed features—to perform **binary classification** (breeding vs. non-breeding) based on whether the recorded DipCount exceeds a threshold (0.1). The deep learning model is implemented with TensorFlow using a hybrid architecture (CNN for sequential weather data and fully connected layers for static site features). **Soon, the TensorFlow model will be replaced with a manually implemented cuDNN version for improved performance/control.**

## Project Structure

- **`data_1.csv`**  
  Contains raw site inspection data downloaded from MMCD.org.

- **`dataManager.py`**  
  Aggregates raw site data (`data_1.csv`) with weather data fetched from the worldWeatherAPI. It generates an augmented CSV (`combined_data.csv`) where for each recorded date additional rows are synthesized to represent the recorded day along with its preceding four days of weather.

- **`combined_data.csv`**  
  The output file produced by `dataManager.py` that contains the merged site and weather data. Each row corresponds to a site record (or a synthesized record for a non-recorded day) enriched with weather fields.

- **`combined_data_filtered.csv`**  
  A version of the combined data where unwanted rows (e.g., those with missing weather entries and a "none" action) have been removed using the `remove_excess_data.py` script.

- **`remove_excess_data.py`**  
  Contains functionality to filter out rows that have missing weather fields (e.g., Temperature) and an "none" action. This ensures that only high-quality, useful records are passed on for training and analysis.

- **`update_combined_data.py`**  
  Periodically updates the combined dataset by merging new raw data and weather data, ensuring that duplicates are removed and new entries are appropriately formatted.

- **`model.py`**  
  Implements the deep learning model for binary classification. The design leverages:
  - **A Convolutional Neural Network (CNN)** specifically engineered to process sequential weather data (e.g., Temperature, Precipitation, CloudCoverage) split into a five-day sequence.
  - **Dense (Fully Connected) Layers** that integrate static site features (e.g., WetlandType, SiteID, categorical indicators) with learned weather patterns.
  - **Output Layer:** A single sigmoid neuron that outputs the probability of a site breeding beyond the 0.1 threshold.

- **`display.py`**  
  Utilizes matplotlib to visualize training performance and various data graphs, such as validation loss curves and accuracy trends.

## Deep Learning Model Overview

The current deep learning model (implemented in `model.py`) follows these key steps:

1. **Data Ingestion & Preprocessing:**  
   The function `load_and_preprocess_data` reads `combined_data.csv` and separates features into:
   - **Weather Data:** A matrix sampled over five consecutive days (e.g., including Temperature, Precipitation, CloudCoverage, and extracted date features) reshaped appropriately for CNN input.
   - **Site Features:** The remaining static attributes (after one-hot encoding and scaling).

2. **Model Architecture:**  
   - **CNN Branch (Weather Data):**  
     A `Conv1D` layer (with 16 filters and a kernel size of 3) scans across the five-day weather sequence. Its output is then flattened into a 1D vector capturing temporal trends.
   - **Site Branch:**  
     The remaining site-specific features feed directly into the network.
   - **Concatenation:**  
     The flattened weather features and the site data are concatenated, allowing the model to learn the interplay between dynamic weather patterns and static site conditions.
   - **Dense Layers:**  
     Multiple fully connected layers (e.g., with 128, 64, and 32 neurons) further process the combined features.
   - **Output:**  
     A final dense layer with a sigmoid activation outputs a probability for binary classification (breeding vs. non-breeding).

3. **Training and Evaluation:**  
   The training process employs common best practices including data splitting (train, validation, test) and loss/accuracy monitoring. Graphs plotted by `display.py` help visualize how the model is learning over epochs.

---

## Usage

1. **Prepare the Data:**
   - Ensure that `data_1.csv` and other raw source data are in place.
   - Run `dataManager.py` to generate or update `combined_data.csv` by merging raw data with weather data.
   - Optionally, run `remove_excess_data.py` to filter `combined_data.csv` into `combined_data_filtered.csv`.

2. **Train the Model:**
   - Execute `model.py` to load the filtered data, train the deep learning model, and evaluate classification accuracy.
   - Monitor training progress with the graphs produced by `display.py`.

3. **Display and Debug:**
   - Use `display.py` to generate visualizations of the training history and to inspect data distributions.

---

## Dependencies

- Python 3.x  
- [Pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [TensorFlow](https://www.tensorflow.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- [matplotlib](https://matplotlib.org/)  
- cuDNN (planned for future implementation)

---

## Future Work

- **Manual cuDNN Implementation:**  
  The current deep learning model uses TensorFlow. A future enhancement is to replace TensorFlow with a low-level cuDNN implementation for improved performance and greater control over the model architecture.
  
- **Enhanced Data Fusion:**  
  Further optimization of the data merging process will ensure minimal redundancy and faster preprocessing when incorporating weather data.

- **Model Tuning and Explainability:**  
  Additional hyperparameter tuning and model explainability tools will be integrated to help better understand which factors most influence mosquito breeding patterns.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Happy coding and may your deep learning model catch every mosquito breeding site!
