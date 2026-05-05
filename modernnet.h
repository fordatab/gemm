/*
 * Modern CNN for Fashion-MNIST dataset
 * VGG-style architecture with 3x3 convolutions
 *
 * Architecture:
 * Block 1: Conv 3x3 (1->16) -> ReLU -> MaxPool 2x2
 * Block 2: Conv 3x3 (16->32) -> ReLU -> MaxPool 2x2
 * Block 3: Conv 3x3 (32->64) -> ReLU -> MaxPool 2x2
 * Block 4: Conv 3x3 (64->128) -> ReLU -> MaxPool 2x2
 * FC (1152->128) -> ReLU
 * FC (128->10) -> Softmax
 */

#ifndef MODERNNET_H_
#define MODERNNET_H_

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <cstdlib>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/conv_cust.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

// Create modern VGG-style network for CPU inference
Network createModernNet_CPU();

// Create modern VGG-style network for CUDA inference
Network createModernNet_CUDA();

#endif  // MODERNNET_H_
