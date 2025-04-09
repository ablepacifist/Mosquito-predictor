#ifndef TRAIN_EVALUATE_H
#define TRAIN_EVALUATE_H

// This function wraps model creation, training, and evaluation.
void train_and_evaluate(float* X_weather_train, float* X_site_train, float* y_train,
                        int num_train_samples,
                        float* X_weather_val, float* X_site_val, float* y_val,
                        int num_val_samples,
                        float* X_weather_test, float* X_site_test, float* y_test,
                        int num_test_samples,
                        int epochs, int batch_size);

#endif // TRAIN_EVALUATE_H
