import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Breeding'] = (data['DipCount'] > 0.1).astype(int)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data['DateOrdinal'] = data['Date'].apply(lambda x: x.toordinal())
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Year'] = data['Date'].dt.year
    data.drop(columns=['Date'], inplace=True)
    le_site = LabelEncoder()
    data['SiteID'] = le_site.fit_transform(data['SiteID'])
    categorical_cols = ['MosquitoRank', 'CattailMosquito', 'CulexFound', 'TreatedBy', 'Action', 'Material']
    for col in categorical_cols:
        data[col] = data[col].fillna('Missing')
    data = pd.get_dummies(data, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
    remaining_obj_cols = data.select_dtypes(include=['object']).columns.tolist()
    if remaining_obj_cols:
        data = pd.get_dummies(data, columns=remaining_obj_cols, drop_first=False)
    numeric_cols = ['WetlandType', 'Wetness', 'DipCount', 'Temperature', 'Precipitation', 
                    'CloudCoverage', 'DateOrdinal', 'Month', 'Day', 'Year']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    # Split weather and site data
    weather_cols = ['Temperature', 'Precipitation', 'CloudCoverage', 'DateOrdinal', 'Month', 'Day', 'Year']
    site_cols = [col for col in data.columns if col not in ['Breeding'] + weather_cols]

    X_weather = data[weather_cols].values
    X_site = data[site_cols].values
    y = data['Breeding'].values

    # Scale weather data
    weather_scaler = StandardScaler()
    X_weather = weather_scaler.fit_transform(X_weather)
    
    # Scale site data
    site_scaler = StandardScaler()
    X_site = site_scaler.fit_transform(X_site)

    y = to_categorical(y)
    
    # Reshape weather data for CNN
    X_weather = X_weather.reshape(X_weather.shape[0], len(weather_cols), 1, 1)

    X_weather_train, X_weather_temp, y_train, y_temp = train_test_split(X_weather, y, test_size=0.3, random_state=42)
    X_weather_val, X_weather_test, y_val, y_test = train_test_split(X_weather_temp, y_temp, test_size=0.5, random_state=42)

    X_site_train, X_site_temp, _, _ = train_test_split(X_site, y, test_size=0.3, random_state=42)
    X_site_val, X_site_test, _, _ = train_test_split(X_site_temp, y_temp, test_size=0.5, random_state=42)

    return X_weather_train, X_site_train, y_train, X_weather_val, X_site_val, y_val, X_weather_test, X_site_test, y_test

def build_cnn_model(weather_input_shape, site_input_shape, num_classes):
    # Weather CNN branch
    weather_input = Input(shape=weather_input_shape)
    # Corrected Kernel size
    weather_conv = Conv2D(16, (3, 1), activation='relu')(weather_input)
    weather_flat = Flatten()(weather_conv)
    weather_dense = Dense(64, activation='relu')(weather_flat)

    # Site data branch
    site_input = Input(shape=site_input_shape)
    site_dense = Dense(64, activation='relu')(site_input)
    
    # Concatenate branches
    combined = concatenate([weather_dense, site_dense])
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(combined)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    dense4 = Dense(16, activation='relu')(dense3)
    output = Dense(num_classes, activation='softmax')(dense4)
    
    model = Model(inputs=[weather_input, site_input], outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(X_weather_train, X_site_train, y_train, X_weather_val, X_site_val, y_val, X_weather_test, X_site_test, y_test):
    weather_input_shape = X_weather_train.shape[1:]
    site_input_shape = X_site_train.shape[1:]
    num_classes = y_train.shape[1]

    model = build_cnn_model(weather_input_shape, site_input_shape, num_classes)
    history = model.fit([X_weather_train, X_site_train], y_train, validation_data=([X_weather_val, X_site_val], y_val), epochs=25, batch_size=128, verbose=1)
    test_loss, test_accuracy = model.evaluate([X_weather_test, X_site_test], y_test, verbose=0)
    return history, test_accuracy

def plot_validation_loss(history):
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def print_first_10_data_points(X_weather, X_site):
    print("First 10 data points for Weather Branch:")
    for i in range(min(10, len(X_weather))):
        print(f"Data Point {i + 1}: {X_weather[i]}")

    print("\nFirst 10 data points for Site Branch:")
    for i in range(min(10, len(X_site))):
        print(f"Data Point {i + 1}: {X_site[i]}")

file_path = 'combined_data.csv'
X_weather_train, X_site_train, y_train, X_weather_val, X_site_val, y_val, X_weather_test, X_site_test, y_test = load_and_preprocess_data(file_path)
print_first_10_data_points(X_weather_train, X_site_train)
print("begin training model")
history, test_accuracy = train_and_evaluate(X_weather_train, X_site_train, y_train, X_weather_val, X_site_val, y_val, X_weather_test, X_site_test, y_test)

plot_validation_loss(history)

print(f"Test Accuracy: {test_accuracy:.4f}")