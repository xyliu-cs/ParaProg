#include "simple_ml_openacc.hpp"

void matrix_dot_openacc(const float *A, const float *B,
                        float *C, size_t M, size_t K, size_t N)
{
    // Initialize the elements of matrix C to zero
    memset(C, 0, M * N * sizeof(float));
    #pragma acc data copyin(A[0:M*K], B[0:K*N]) copyout(C[0:M*N]) 
    {
        // OpenACC parallel loop directive
        #pragma acc parallel loop collapse(2) 
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                // OpenACC loop directive for vectorization
                #pragma acc loop reduction(+:sum) 
                for (size_t k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

void A_transpose_dot_B_openacc(const float *A, const float *B, float *C, size_t K, size_t M, size_t N)
{
    memset(C, 0, M * N * sizeof(float));
    #pragma acc data copyin(A[0:K*M], B[0:K*N]) copyout(C[0:M*N])
    {
    // OpenACC parallel loop directive
    # pragma acc region
    {
        #pragma acc loop independent vector(32)
        for (size_t i = 0; i < M; i++) {
        #pragma acc loop independent vector(32)
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += A[k * M + i] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    }
}

void A_dot_B_transpose_openacc(const float *A, const float *B, float *C, size_t M, size_t K, size_t N)
{
    // Initialize the elements of matrix C to zero
    memset(C, 0, M * N * sizeof(float));

    // Manage data movement for A, B, and C
    #pragma acc data copyin(A[0:M*K], B[0:N*K]) copyout(C[0:M*N])
    {
        // Parallelize the outer two loops
        #pragma acc parallel loop collapse(2)
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;

                // Inner loop with reduction to avoid race conditions
                #pragma acc loop reduction(+:sum)
                for (size_t k = 0; k < K; ++k) {
                    sum += A[m * K + k] * B[n * K + k];
                }

                C[m * N + n] = sum;
            }
        }
    }
}

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    for (size_t i = 0; i < m * n; ++i) {
        A[i] -= B[i];
    }
}

void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    for (size_t i = 0; i < m * n; ++i) {
        C[i] *= scalar;
    }
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    for (size_t i = 0; i < m * n; ++i) {
        C[i] /= scalar;
    }
}

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    // Iterate over each row
    #pragma acc data copy(C[0:m*n])
    {
        #pragma acc parallel loop
        for (size_t i = 0; i < m; ++i) {
            float exp_sum = 0.0;

            #pragma acc loop reduction(+:exp_sum)
            // First pass: compute the exponentials and sum them
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
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t k)
{
    // First, set all elements in Y to 0
    memset(Y, 0, m * k * sizeof(float));

    // Iterate over each element of y
    for (size_t i = 0; i < m; ++i) {
        unsigned char true_class = y[i];
        // Check if the label is within the valid range
        if (true_class < k) {
            // Set the corresponding element in Y to 1
            Y[i * k + true_class] = 1.0f;
        }
    }
}


float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    float total_loss_sum = 0.0;
    #pragma acc data copyin(result[0:images_num*num_classes], labels_array[0:images_num])
    {
        #pragma acc parallel loop reduction(+:total_loss_sum)
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
    }
    return total_loss_sum / images_num;
}

float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
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

void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        A[i] *= B[i];
    }
}


void relu_openacc(float *matrix, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (matrix[i] < 0) {
            matrix[i] = 0;
        }
    }
}


void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch)
{
    // Temporary storage for a batch of X, logits, one-hot labels, and gradients
    // float* X_b = new float[batch * n]();
    float* Z = new float[batch * k];
    float* Y = new float[batch * k];
    float* gradients = new float[n * k];


    for (size_t i = 0; i < (m - batch); i += batch) {
        // Determine the actual batch size (might be smaller at the end of the dataset)
        // Copy the current batch of X into X_batch
        auto X_b = X + i * n;
        // memcpy(X_b, X + i * n, actual_batch_size * n * sizeof(float));

        // Compute logits: logits = X_b.dot(theta)
        matrix_dot_openacc(X_b, theta, Z, batch, n, k);

        // Softmax normalization (Z: batch x k)
        // Z = np.exp(logits) / np.sum(np.exp(logits), axis=1)[:, None]
        matrix_softmax_normalize_openacc(Z, batch, k);

        // Convert y to one-hot encoding (Y: batch x k)
        // Notice that y also needs to shift by batch (+i)
        vector_to_one_hot_matrix_openacc(y + i, Y, batch, k);

        // Compute gradients: gradients = X_b.T.dot(Z - Y) / batch * lr
        // Compute gradients #1: Z - Y (batch x k)
        matrix_minus_openacc(Z, Y, batch, k);

        // Compute gradients #2: gradients = X_b.T.dot(Z - Y)
        // n x batch  dot  batch x k 
        A_transpose_dot_B_openacc(X_b, Z, gradients, batch, n, k);

        // Compute gradients #3 gradients = gradients * lr / batch_size
        float rate = lr / batch;
        matrix_mul_scalar_openacc(gradients, rate, n, k);

        // Update theta: theta -= gradients
        matrix_minus_openacc(theta, gradients, n, k);
    }

    // Deallocate memory
    // delete[] X_b;
    delete[] Z;
    delete[] Y;
    delete[] gradients;
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE  
        // Run a single epoch of softmax regression
        softmax_regression_epoch_openacc(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch);

        // C = X @ theta
        matrix_dot_openacc(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
        matrix_dot_openacc(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);

        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
   // END YOUR CODE
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}


void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
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
        matrix_dot_openacc(X_b, W1, Z1, current_batch_size, n, l);
        relu_openacc(Z1, current_batch_size * l);

        matrix_dot_openacc(Z1, W2, Z2, current_batch_size, l, k);
        matrix_softmax_normalize_openacc(Z2, current_batch_size, k); // Using the provided softmax function

        // Backward pass
        // vector_to_one_hot_matrix(y + i, Y, current_batch_size, k);
        auto Y_b = Y + i * k;
        matrix_minus_openacc(Z2, Y_b, current_batch_size, k);

        A_dot_B_transpose_openacc(Z2, W2, G1, current_batch_size, k, l);

        // Applying ReLU gradient: G1 *= (Z1 > 0)
        for (size_t j = 0; j < current_batch_size * l; ++j) {
            G1[j] *= (Z1[j] > 0);
        }

        A_transpose_dot_B_openacc(X_b, G1, W1_l, current_batch_size, n, l);
        matrix_mul_scalar_openacc(W1_l, lr / current_batch_size, n, l);

        A_transpose_dot_B_openacc(Z1, Z2, W2_l, current_batch_size, l, k);
        matrix_mul_scalar_openacc(W2_l, lr / current_batch_size, l, k);

        // Update weights
        matrix_minus_openacc(W1, W1_l, n, l);
        matrix_minus_openacc(W2, W2_l, l, k);
    }

    delete[] Z1;
    delete[] Z2;
    delete[] Y;
    delete[] G1;
    delete[] W1_l;
    delete[] W2_l;
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
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
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // result = relu(X @ W1) @ W2
        // Perform one epoch of training
        nn_epoch_openacc(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, lr, batch);

        // tmp for X dot W1
        float* X_dot_W1 = new float [train_data->images_num * hidden_dim];
        // Forward pass for the training dataset
        matrix_dot_openacc(train_data->images_matrix, W1, X_dot_W1, train_data->images_num, train_data->input_dim, hidden_dim);
        relu_openacc(X_dot_W1, train_data->images_num * hidden_dim);
        matrix_dot_openacc(X_dot_W1, W2, train_result, train_data->images_num, hidden_dim, num_classes);
        // matrix_softmax_normalize(train_result, train_data->images_num, num_classes);

        // Forward pass for the testing dataset
        float* X_dot_W1_test = new float [test_data->images_num * hidden_dim];
        matrix_dot_openacc(test_data->images_matrix, W1, X_dot_W1_test, test_data->images_num, test_data->input_dim, hidden_dim);
        relu_openacc(X_dot_W1_test, test_data->images_num * hidden_dim);
        matrix_dot_openacc(X_dot_W1_test, W2, test_result, test_data->images_num, hidden_dim, num_classes);


        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
}
