#include "simple_ml_ext.hpp"

DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim)
{
    images_matrix = new float[images_num * input_dim];
    labels_array = new unsigned char[images_num];
}

DataSet::~DataSet()
{
    delete[] images_matrix;
    delete[] labels_array;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename)
{
    std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
    uint32_t magic_num, images_num, rows_num, cols_num;

    images_file.read(reinterpret_cast<char *>(&magic_num), 4);
    labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

    images_file.read(reinterpret_cast<char *>(&images_num), 4);
    labels_file.read(reinterpret_cast<char *>(&images_num), 4);
    images_num = swap_endian(images_num);

    images_file.read(reinterpret_cast<char *>(&rows_num), 4);
    rows_num = swap_endian(rows_num);
    images_file.read(reinterpret_cast<char *>(&cols_num), 4);
    cols_num = swap_endian(cols_num);

    DataSet *dataset = new DataSet(images_num, rows_num * cols_num);

    labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);
    unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
    images_file.read(reinterpret_cast<char *>(pixels), images_num * rows_num * cols_num);
    for (size_t i = 0; i < images_num * rows_num * cols_num; i++)
    {
        dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;
    }

    delete[] pixels;

    return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(const float *A, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size M * K
 *     B (const float*): Matrix of size K * N
 *     C (float*): Matrix of size M * N
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t M, size_t K, size_t N)
{
    // Initialize the elements of matrix C to zero
    memset(C, 0, M * N * sizeof(float));
    // loop interchange
    for (size_t i = 0; i < M; i++) {    
        for (size_t k = 0; k < K; k++) {
            const float a_element = A[i * K + k];
            for (size_t j = 0; j < N; j++) {
                C[i * N + j] += a_element * B[k * N + j];
            }
        }
    }
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size K * M
 *     B (const float*): Matrix of size K * N
 *     C (float*): Matrix of size M * N
 **/
void A_transpose_dot_B(const float *A, const float *B, float *C, size_t K, size_t M, size_t N)
{
    // Initialize the elements of matrix C to zero
    memset(C, 0, M * N * sizeof(float));
    // loop interchange
    for (size_t k = 0; k < K; k++) {
        for (size_t i = 0; i < M; i++) {    
            const float a_trans_element = A[k * M + i];
            for (size_t j = 0; j < N; j++) {
                C[i * N + j] += a_trans_element * B[k * N + j];
            }
        }
    }
}

/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size M * K
 *     B (const float*): Matrix of size N * K
 *     C (float*): Matrix of size M * N
 **/
void A_dot_B_transpose(const float *A, const float *B, float *C, size_t M, size_t K, size_t N)
{
    // Initialize the elements of matrix C to zero
    memset(C, 0, M * N * sizeof(float));
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t m, size_t n)
{
    for (size_t i = 0; i < m * n; ++i) {
        A[i] -= B[i];
    }
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t m, size_t n)
{
    for (size_t i = 0; i < m * n; ++i) {
        C[i] *= scalar;
    }
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t m, size_t n)
{
    for (size_t i = 0; i < m * n; ++i) {
        C[i] /= scalar;
    }
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t m, size_t n)
{
    // Iterate over each row
    for (size_t i = 0; i < m; ++i) {
        // First pass: compute the exponentials and sum them
        float exp_sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] = expf(C[i * n + j]); 
            exp_sum += C[i * n + j];
        }

        // Second pass: divide by the sum to normalize
        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] /= exp_sum;
        }
    }
}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     y (unsigned char *): vector of size m * 1
 *     Y (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m, size_t n)
{
    // First, set all elements in Y to 0
    memset(Y, 0, m * n * sizeof(float));

    // Iterate over each element of y
    for (size_t i = 0; i < m; ++i) {
        // unsigned char true_class = y[i];
        // // Check if the label is within the valid range
        // if (true_class < n) {
        //     // Set the corresponding element in Y to 1
        //     Y[i * n + true_class] = 1.0f;
        // }
        Y[i * n + y[i]] = 1.0f;
    }
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // 1. Non-scaling Version 
    // float total_loss_sum = 0.0;
    // for (size_t i = 0; i < images_num; ++i) {
    //     float sum_exp = 0.0;
    //     for (size_t j = 0; j < num_classes; ++j) {
    //         sum_exp += expf(result[i * num_classes + j]);
    //     }
    //     unsigned char true_class = labels_array[i];
    //     total_loss_sum += logf(sum_exp) - result[i * num_classes + true_class];
    // }
    // return total_loss_sum / images_num;
    
    // 2. Numerical stabilization Version
    float total_loss_sum = 0.0;
    for (size_t i = 0; i < images_num; ++i) {
        float max_val = result[i * num_classes];
        for (size_t j = 1; j < num_classes; ++j) {
            if (result[i * num_classes + j] > max_val) {
                max_val = result[i * num_classes + j];
            }
        }

        float sum_exp = 0.0;
        for (size_t j = 0; j < num_classes; ++j) {
            sum_exp += expf(result[i * num_classes + j] - max_val);  // Subtract max_val for stabilization
        }

        unsigned char true_class = labels_array[i];
        total_loss_sum += logf(sum_exp) + max_val - result[i * num_classes + true_class];  // Add max_val back to the loss
    }
    return total_loss_sum / images_num;
}

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{ 
    size_t error_count = 0;
    // std::cout << "mean_err: sample result[-1] = " << result[-1] << std::endl;

    for (size_t i = 0; i < images_num; ++i) {
        // Find the index of the max logit for the current example
        size_t max_logit_index = 0;
        float max_logit = result[i * num_classes];

        for (size_t j = 1; j < num_classes; ++j) {
            if (result[i * num_classes + j] > max_logit) {
                max_logit = result[i * num_classes + j];
                max_logit_index = j;
            }
        }

        // Compare the predicted class (max_logit_index) with the actual class (labels_array[i])
        if (max_logit_index != labels_array[i]) {
            // std::cout << "max_logit_index = " << max_logit_index << std::endl;
            error_count++;
        }
    }

    // Calculate the average error
    return static_cast<float>(error_count) / images_num;
}

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] *= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        A[i] *= B[i];
    }
}

/**
 * ReLU operation
 **/
void relu(float *matrix, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (matrix[i] < 0) {
            matrix[i] = 0;
        }
    }
}






/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the logits and gradients.
 *
 * Args:
 *     X (const float *): pointer to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): pointer to y data, of size m
 *     theta (float *): pointer to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (int): size of SGD batch
 *
 * Returns:
 *     (None)
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    // Temporary storage for a batch of X, logits, one-hot labels, and gradients
    // float* X_b = new float[batch * n]();
    float* Z = new float[batch * k];
    // float* Y = new float[batch * k];
    float* Y = new float[m * k];
    float* gradients = new float[n * k];
    vector_to_one_hot_matrix(y, Y, m, k);


    for (size_t i = 0; i < (m - batch); i += batch) {
        // Determine the actual batch size (might be smaller at the end of the dataset)
        // Copy the current batch of X into X_batch
        auto X_b = X + i * n;
        // memcpy(X_b, X + i * n, actual_batch_size * n * sizeof(float));

        // Compute logits: logits = X_b.dot(theta)
        matrix_dot(X_b, theta, Z, batch, n, k);

        // Softmax normalization (Z: batch x k)
        // Z = np.exp(logits) / np.sum(np.exp(logits), axis=1)[:, None]
        matrix_softmax_normalize(Z, batch, k);

        // Convert y to one-hot encoding (Y: batch x k)
        // Notice that y also needs to shift by batch (+i)
        auto Y_b = Y + i * k;
        // vector_to_one_hot_matrix(y + i, Y, batch, k);

        // Compute gradients: gradients = X_b.T.dot(Z - Y) / batch * lr
        // Compute gradients #1: Z - Y (batch x k)
        matrix_minus(Z, Y_b, batch, k);

        // Compute gradients #2: gradients = X_b.T.dot(Z - Y)
        // n x batch  dot  batch x k 
        A_transpose_dot_B(X_b, Z, gradients, batch, n, k);

        // Compute gradients #3 gradients = gradients * lr / batch_size
        float rate = lr / batch;
        matrix_mul_scalar(gradients, rate, n, k);

        // Update theta: theta -= gradients
        matrix_minus(theta, gradients, n, k);
    }

    // Deallocate memory
    // delete[] X_b;
    delete[] Z;
    delete[] Y;
    delete[] gradients;
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE
        // Run a single epoch of softmax regression
        softmax_regression_epoch_cpp(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch);

        // C = X @ theta
        matrix_dot(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
        matrix_dot(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);
        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}


/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    logits = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD batch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    float *Z1 = new float[batch * l]();
    float *Z2 = new float[batch * k]();
    float *Y = new float[m * k]();
    float *G1 = new float[batch * l]();
    float *W1_l = new float[n * l]();
    float *W2_l = new float[l * k]();
    vector_to_one_hot_matrix(y, Y, m, k);

    for (size_t i = 0; i < m; i += batch) {
        size_t current_batch_size = std::min(batch, m - i);

        // Forward pass
        auto X_b = X + i * n;
        matrix_dot(X_b, W1, Z1, current_batch_size, n, l);
        relu(Z1, current_batch_size * l);

        matrix_dot(Z1, W2, Z2, current_batch_size, l, k);
        matrix_softmax_normalize(Z2, current_batch_size, k); // Using the provided softmax function

        // Backward pass
        // vector_to_one_hot_matrix(y + i, Y, current_batch_size, k);
        auto Y_b = Y + i * k;
        matrix_minus(Z2, Y_b, current_batch_size, k);

        A_dot_B_transpose(Z2, W2, G1, current_batch_size, k, l);

        // Applying ReLU gradient: G1 *= (Z1 > 0)
        for (size_t j = 0; j < current_batch_size * l; ++j) {
            G1[j] *= (Z1[j] > 0);
        }

        A_transpose_dot_B(X_b, G1, W1_l, current_batch_size, n, l);
        matrix_mul_scalar(W1_l, lr / current_batch_size, n, l);

        A_transpose_dot_B(Z1, Z2, W2_l, current_batch_size, l, k);
        matrix_mul_scalar(W2_l, lr / current_batch_size, l, k);

        // Update weights
        matrix_minus(W1, W1_l, n, l);
        matrix_minus(W2, W2_l, l, k);
    }

    delete[] Z1;
    delete[] Z2;
    delete[] Y;
    delete[] G1;
    delete[] W1_l;
    delete[] W2_l;
}

/**
 *Example function to fully train a nn classifier
 **/
void train_nn(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
    float *train_result = new float[train_data->images_num * num_classes]();
    float *test_result = new float[test_data->images_num * num_classes]();
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE
        // result = relu(X @ W1) @ W2
        // Perform one epoch of training
        nn_epoch_cpp(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, lr, batch);

        // tmp for X dot W1
        float* X_dot_W1 = new float [train_data->images_num * hidden_dim];
        // Forward pass for the training dataset
        matrix_dot(train_data->images_matrix, W1, X_dot_W1, train_data->images_num, train_data->input_dim, hidden_dim);
        relu(X_dot_W1, train_data->images_num * hidden_dim);
        matrix_dot(X_dot_W1, W2, train_result, train_data->images_num, hidden_dim, num_classes);
        // matrix_softmax_normalize(train_result, train_data->images_num, num_classes);

        // Forward pass for the testing dataset
        float* X_dot_W1_test = new float [test_data->images_num * hidden_dim];
        matrix_dot(test_data->images_matrix, W1, X_dot_W1_test, test_data->images_num, test_data->input_dim, hidden_dim);
        relu(X_dot_W1_test, test_data->images_num * hidden_dim);
        matrix_dot(X_dot_W1_test, W2, test_result, test_data->images_num, hidden_dim, num_classes);
        // matrix_softmax_normalize(test_result, test_data->images_num, num_classes);
        
        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
        
        delete[] X_dot_W1;
        delete[] X_dot_W1_test;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
