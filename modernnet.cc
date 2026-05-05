/*
 * Modern CNN for Fashion-MNIST dataset
 * VGG-style architecture with 3x3 convolutions
 */

#include "modernnet.h"

/*
 * Dimension calculation for 86x86 input:
 *
 * Block 1: Conv 3x3 (1->16): 84x84, MaxPool 2x2: 42x42
 * Block 2: Conv 3x3 (16->32): 40x40, MaxPool 2x2: 20x20
 * Block 3: Conv 3x3 (32->64): 18x18, MaxPool 2x2: 9x9
 * Block 4: Conv 3x3 (64->128): 7x7, MaxPool 2x2: 3x3
 * Flatten: 3x3x128 = 1152
 * FC: 1152->128->10
 */

Network createModernNet_CPU()
{
    Network dnn;

    // Block 1: Input 86x86x1 -> 42x42x16
    Layer* conv1 = new Conv(1, 86, 86, 16, 3, 3);      // -> 84x84x16
    Layer* relu1 = new ReLU;
    Layer* pool1 = new MaxPooling(16, 84, 84, 2, 2, 2); // -> 42x42x16

    // Block 2: 42x42x16 -> 20x20x32
    Layer* conv2 = new Conv(16, 42, 42, 32, 3, 3);     // -> 40x40x32
    Layer* relu2 = new ReLU;
    Layer* pool2 = new MaxPooling(32, 40, 40, 2, 2, 2); // -> 20x20x32

    // Block 3: 20x20x32 -> 9x9x64
    Layer* conv3 = new Conv(32, 20, 20, 64, 3, 3);     // -> 18x18x64
    Layer* relu3 = new ReLU;
    Layer* pool3 = new MaxPooling(64, 18, 18, 2, 2, 2); // -> 9x9x64

    // Block 4: 9x9x64 -> 3x3x128
    Layer* conv4 = new Conv(64, 9, 9, 128, 3, 3);      // -> 7x7x128
    Layer* relu4 = new ReLU;
    Layer* pool4 = new MaxPooling(128, 7, 7, 2, 2, 2); // -> 3x3x128

    // Classifier: 1152 -> 128 -> 10
    Layer* fc1 = new FullyConnected(pool4->output_dim(), 128);
    Layer* relu5 = new ReLU;
    Layer* fc2 = new FullyConnected(128, 10);
    Layer* softmax = new Softmax;

    // Build network
    dnn.add_layer(conv1);
    dnn.add_layer(relu1);
    dnn.add_layer(pool1);

    dnn.add_layer(conv2);
    dnn.add_layer(relu2);
    dnn.add_layer(pool2);

    dnn.add_layer(conv3);
    dnn.add_layer(relu3);
    dnn.add_layer(pool3);

    dnn.add_layer(conv4);
    dnn.add_layer(relu4);
    dnn.add_layer(pool4);

    dnn.add_layer(fc1);
    dnn.add_layer(relu5);
    dnn.add_layer(fc2);
    dnn.add_layer(softmax);

    // Loss
    Loss* loss = new CrossEntropy;
    dnn.add_loss(loss);

    return dnn;
}

Network createModernNet_CUDA()
{
    Network dnn;

    // Block 1: Input 86x86x1 -> 42x42x16
    Layer* conv1 = new Conv_Custom(1, 86, 86, 16, 3, 3);  // -> 84x84x16
    Layer* relu1 = new ReLU;
    Layer* pool1 = new MaxPooling(16, 84, 84, 2, 2, 2);   // -> 42x42x16

    // Block 2: 42x42x16 -> 20x20x32
    Layer* conv2 = new Conv_Custom(16, 42, 42, 32, 3, 3); // -> 40x40x32
    Layer* relu2 = new ReLU;
    Layer* pool2 = new MaxPooling(32, 40, 40, 2, 2, 2);   // -> 20x20x32

    // Block 3: 20x20x32 -> 9x9x64
    Layer* conv3 = new Conv_Custom(32, 20, 20, 64, 3, 3); // -> 18x18x64
    Layer* relu3 = new ReLU;
    Layer* pool3 = new MaxPooling(64, 18, 18, 2, 2, 2);   // -> 9x9x64

    // Block 4: 9x9x64 -> 3x3x128
    Layer* conv4 = new Conv_Custom(64, 9, 9, 128, 3, 3);  // -> 7x7x128
    Layer* relu4 = new ReLU;
    Layer* pool4 = new MaxPooling(128, 7, 7, 2, 2, 2);    // -> 3x3x128

    // Classifier: 1152 -> 128 -> 10
    Layer* fc1 = new FullyConnected(pool4->output_dim(), 128);
    Layer* relu5 = new ReLU;
    Layer* fc2 = new FullyConnected(128, 10);
    Layer* softmax = new Softmax;

    // Build network
    dnn.add_layer(conv1);
    dnn.add_layer(relu1);
    dnn.add_layer(pool1);

    dnn.add_layer(conv2);
    dnn.add_layer(relu2);
    dnn.add_layer(pool2);

    dnn.add_layer(conv3);
    dnn.add_layer(relu3);
    dnn.add_layer(pool3);

    dnn.add_layer(conv4);
    dnn.add_layer(relu4);
    dnn.add_layer(pool4);

    dnn.add_layer(fc1);
    dnn.add_layer(relu5);
    dnn.add_layer(fc2);
    dnn.add_layer(softmax);

    // Loss
    Loss* loss = new CrossEntropy;
    dnn.add_loss(loss);

    return dnn;
}
