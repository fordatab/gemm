/*
 * Main program for Modern CNN on Fashion-MNIST
 * Supports both training and inference modes
 */

#include "modernnet.h"
#include <cstring>

void train(int epochs, int batch_size, float learning_rate) {
    std::cout << "Loading Fashion-MNIST training data..." << std::endl;
    MNIST dataset("./data/");
    dataset.read();
    std::cout << "Done. Training samples: " << dataset.train_data.cols() << std::endl;

    std::cout << "Creating Modern VGG-style network..." << std::endl;
    // Hybrid training: GPU forward (Conv_Custom), CPU backward/update.
    Conv_Custom::verbose = false;  // silence per-conv timing spam during training
    Network dnn = createModernNet_CUDA();
    std::cout << "Done" << std::endl;

    // Optimizer
    SGD opt(learning_rate, 5e-4, 0.9, true);

    // Training loop
    int n_batch = dataset.train_data.cols() / batch_size;

    // CSV logs for plotting later
    std::ofstream loss_log("./build/training_loss.csv");
    loss_log << "epoch,batch,global_step,loss\n";
    std::ofstream acc_log("./build/epoch_accuracy.csv");
    acc_log << "epoch,test_accuracy\n";

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_data(dataset.train_data, dataset.train_labels);

        float total_loss = 0;
        for (int batch = 0; batch < n_batch; batch++) {
            int start = batch * batch_size;
            Matrix batch_data = dataset.train_data.block(0, start,
                                    dataset.train_data.rows(), batch_size);
            Matrix batch_labels = dataset.train_labels.block(0, start,
                                    dataset.train_labels.rows(), batch_size);
            // Loss expects one-hot targets (n_classes x batch), but labels are
            // stored as raw class indices (1 x batch).
            Matrix batch_onehot = one_hot_encode(batch_labels, 10);

            dnn.forward(batch_data);
            dnn.backward(batch_data, batch_onehot);
            dnn.update(opt);

            float cur_loss = dnn.get_loss();
            total_loss += cur_loss;

            // Log instantaneous per-batch loss for plotting
            loss_log << epoch + 1 << "," << batch + 1 << ","
                     << epoch * n_batch + batch << "," << cur_loss << "\n";

            if (batch == 0 || (batch + 1) % 10 == 0) {
                std::cout << "\rEpoch " << epoch + 1 << "/" << epochs
                          << " | Batch " << batch + 1 << "/" << n_batch
                          << " | Loss: " << total_loss / (batch + 1) << std::flush;
            }
        }

        // Validation accuracy.
        // Forward the test set in chunks instead of all 10k at once: the GPU
        // conv path allocates im2col/output buffers sized for the whole batch,
        // and a single 10k-wide forward overflows VRAM on the deeper, high-
        // channel conv layers. Chunking keeps each forward within device memory.
        int n_test = dataset.test_data.cols();
        int val_batch = batch_size;
        int correct = 0;
        for (int start = 0; start < n_test; start += val_batch) {
            int cur = (n_test - start < val_batch) ? (n_test - start) : val_batch;
            Matrix chunk = dataset.test_data.block(0, start,
                                dataset.test_data.rows(), cur);
            dnn.forward(chunk);
            const Matrix& preds = dnn.output();
            Matrix chunk_labels = dataset.test_labels.block(0, start, 1, cur);
            for (int i = 0; i < cur; i++) {
                Matrix::Index max_index;
                preds.col(i).maxCoeff(&max_index);
                correct += (int(max_index) == chunk_labels(i));
            }
        }
        float acc = static_cast<float>(correct) / n_test;
        std::cout << "\nEpoch " << epoch + 1 << " complete. Test Accuracy: "
                  << acc * 100 << "%" << std::endl;
        acc_log << epoch + 1 << "," << acc << "\n";
        loss_log.flush();
        acc_log.flush();
    }

    // Save trained weights
    std::cout << "Saving weights to ./build/modern-weights.bin..." << std::endl;
    dnn.save_parameters("./build/modern-weights.bin");
    std::cout << "Done" << std::endl;
}

void inference_cpu(int batch_size) {
    std::cout << "Loading Fashion-MNIST test data..." << std::endl;
    MNIST dataset("./data/");
    dataset.read_test_data(batch_size);
    std::cout << "Done" << std::endl;

    std::cout << "Loading Modern network (CPU)..." << std::endl;
    Network dnn = createModernNet_CPU();

    // Try to load pre-trained weights
    std::ifstream file("./build/modern-weights.bin");
    if (file.good()) {
        file.close();
        dnn.load_parameters("./build/modern-weights.bin");
        std::cout << "Loaded pre-trained weights" << std::endl;
    } else {
        std::cout << "No pre-trained weights found, using random initialization" << std::endl;
    }

    std::cout << "Running inference..." << std::endl;
    dnn.forward(dataset.test_data);
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);

    std::cout << std::endl;
    std::cout << "Test Accuracy: " << acc * 100 << "%" << std::endl;
    std::cout << std::endl;
}

void inference_cuda(int batch_size) {
    std::cout << "Loading Fashion-MNIST test data..." << std::endl;
    MNIST dataset("./data/");
    dataset.read_test_data(batch_size);
    std::cout << "Done" << std::endl;

    std::cout << "Loading Modern network (CUDA)..." << std::endl;
    Network dnn = createModernNet_CUDA();

    // Try to load pre-trained weights
    std::ifstream file("./build/modern-weights.bin");
    if (file.good()) {
        file.close();
        dnn.load_parameters("./build/modern-weights.bin");
        std::cout << "Loaded pre-trained weights" << std::endl;
    } else {
        std::cout << "No pre-trained weights found, using random initialization" << std::endl;
    }

    std::cout << "Running inference..." << std::endl;
    dnn.forward(dataset.test_data);
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);

    std::cout << std::endl;
    std::cout << "Test Accuracy: " << acc * 100 << "%" << std::endl;
    std::cout << std::endl;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [mode] [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  train    Train the network" << std::endl;
    std::cout << "  cpu      Run inference on CPU" << std::endl;
    std::cout << "  cuda     Run inference on CUDA GPU" << std::endl;
    std::cout << std::endl;
    std::cout << "Options for train mode:" << std::endl;
    std::cout << "  --epochs N       Number of epochs (default: 10)" << std::endl;
    std::cout << "  --batch N        Batch size (default: 128)" << std::endl;
    std::cout << "  --lr RATE        Learning rate (default: 0.01)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options for inference modes:" << std::endl;
    std::cout << "  --batch N        Test batch size (default: 1000)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog << " train --epochs 20 --batch 64" << std::endl;
    std::cout << "  " << prog << " cpu --batch 5000" << std::endl;
    std::cout << "  " << prog << " cuda --batch 10000" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    // Default parameters
    int epochs = 10;
    int batch_size = 128;
    float learning_rate = 0.01f;
    int test_batch = 1000;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
            test_batch = batch_size;
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        }
    }

    if (mode == "train") {
        std::cout << "=== Modern VGG-style CNN Training ===" << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Learning rate: " << learning_rate << std::endl;
        std::cout << std::endl;
        train(epochs, batch_size, learning_rate);
    } else if (mode == "cpu") {
        std::cout << "=== Modern VGG-style CNN Inference (CPU) ===" << std::endl;
        std::cout << "Test batch size: " << test_batch << std::endl;
        inference_cpu(test_batch);
    } else if (mode == "cuda") {
        std::cout << "=== Modern VGG-style CNN Inference (CUDA) ===" << std::endl;
        std::cout << "Test batch size: " << test_batch << std::endl;
        inference_cuda(test_batch);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
