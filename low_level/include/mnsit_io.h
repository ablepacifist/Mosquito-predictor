// mnist_io.h
#ifndef MNIST_IO_H
#define MNIST_IO_H


#include <vector>

void loadMNISTImages(const char* filename, std::vector<float>& images, int& num_images);
void loadMNISTLabels(const char* filename, std::vector<float>& labels, int num_images, int num_classes);

// Declare mnistMain with an explicit return type or remove it if unnecessary
int mnistMain();
 #endif // MNIST_IO_H
