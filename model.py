import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load and preprocess the MMCD dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(file_path):
    # 1. Load the dataset.
    data = pd.read_csv(file_path)
    
    # 2. Create a binary target "Breeding": 1 if DipCount > 0.1, 0 otherwise.
    data['Breeding'] = (data['DipCount'] > 0.1).astype(int)
    
    # 3. Process the Date column.
    # Convert Date to datetime and extract numeric components.
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data['DateOrdinal'] = data['Date'].apply(lambda x: x.toordinal())
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Year'] = data['Date'].dt.year
    # Drop the original non-numeric Date since information is preserved.
    data.drop(columns=['Date'], inplace=True)
    
    # 4. Process SiteID.
    # Convert SiteID (string) to numeric using label encoding.
    le_site = LabelEncoder()
    data['SiteID'] = le_site.fit_transform(data['SiteID'])
    
    # 5. Process explicit categorical columns.
    categorical_cols = ['MosquitoRank', 'CattailMosquito', 'CulexFound', 'TreatedBy', 'Action', 'Material']
    for col in categorical_cols:
        data[col] = data[col].fillna('Missing')
    data = pd.get_dummies(data, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
    
    # 6. Extra step: One-hot encode any remaining object-type columns.
    remaining_obj_cols = data.select_dtypes(include=['object']).columns.tolist()
    if remaining_obj_cols:
        data = pd.get_dummies(data, columns=remaining_obj_cols, drop_first=False)
    
    # 7. Handle numeric columns — fill missing values for those that exist.
    numeric_cols = ['WetlandType', 'Wetness', 'DipCount', 'Temperature', 'Precipitation', 
                    'CloudCoverage', 'DateOrdinal', 'Month', 'Day', 'Year']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    # 8. Separate features and the target.
    X = data.drop(columns=['Breeding']).values
    y = data['Breeding'].values

    # 9. Standardize the features.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 10. Convert labels to one-hot encoding.
    y = to_categorical(y)
    
    # 11. Reshape features for CNN input.
    # CNN expects a 4D input in the format: (samples, features, 1, 1) !!!!
    X = X.reshape(X.shape[0], 5, 11, 1)
    
    # 12. Split into training, validation, and test sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# 2. Build the CNN model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Train and evaluate the model
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    input_shape = X_train.shape[1:]  # Shape of a single sample
    num_classes = y_train.shape[1]  # Number of classes

    model = build_cnn_model(input_shape, num_classes)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=128, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return history, test_accuracy

# 4. Plot validation loss
def plot_validation_loss(history):
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main script
file_path = 'combined_data.csv'  # Path to the dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(file_path)

# Train and evaluate the model
print("begin training model")
history, test_accuracy = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

# Plot validation loss
plot_validation_loss(history)

# Report final accuracy
print(f"Test Accuracy: {test_accuracy:.4f}")