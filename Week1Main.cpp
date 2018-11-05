#include <mlpack/core.hpp>
#include <armadillo>							// !!! Must go after mlpack/core.hpp   ref: email from Ryan Curtin. !!!
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>   // For Xavier Iniialisation
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>

#include <h5cpp/all>

#include <vector>
#include <algorithm>
#include <iostream>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

void main()
{
	////////////////////////////////////////////////////////////////////////
	//
	// Convolution Model - Application
	//
	// # Imports 1,080 training images and 120 testing images of dimension
	// # 64x64x3 from .H5 databases and runs a Conv net replicating the 
	// # DeepLearning.ai exercise model. 
	//  
	// # Note:  The Python model in the exercise results in a training
	// #        accuracy of 94.1% and a test accuracy of 78.3%. Nothing
	// #        like this has yet been achieved by me using MLPACK.
	//
	// # Author:  David Armour
	//
	////////////////////////////////////////////////////////////////////////
	arma::mat trainSetX(64 * 64 * 3, 1080);
	arma::mat trainSetY(1, 1080);
	arma::mat testSetX(64 * 64 * 3, 120);
	arma::mat testSetY(1, 120);
	arma::mat trainX, testX, trainY, testY;
	arma::mat classes;

	// Load the Signs Data from the .h5 files. The data needs to be loaded in (64x64 + 64x64 + 64x64) x m  (m = nbr of samples).
	{
		auto fd = h5::open("data/train_signs.h5", H5F_ACC_RDONLY);
		for (unsigned long long i = 0; i < trainSetX.n_cols; ++i)
		{
			std::vector<int> tempCol;   // Temporary used to correctly insert image vectors to the matrix. Needed because data
												 // is not stored in the necessary way for MLPACK Convolutions.

			for (unsigned int k = 0; k < 3; ++k)
			{
				auto in = h5::read<std::vector<int> >(fd, "train_set_x", h5::offset{ i,0,0,k }, h5::count{ 1, 64, 64, 1 }, h5::stride{ testSetX.n_cols,1,1,3 });
				tempCol.insert(std::end(tempCol), std::begin(in), std::end(in));
			}
			trainSetX.col(i) = arma::conv_to<arma::colvec>::from(tempCol);
		}
		trainSetY = arma::mat(h5::read<arma::rowvec>(fd, "train_set_y"));
	}

	{
		auto fd = h5::open("data/test_signs.h5", H5F_ACC_RDONLY);
		for (unsigned long long i = 0; i < testSetX.n_cols; ++i)
		{
			std::vector<int> tempCol;   // Temporary used to correctly insert image vectors to the matrix. Needed because data
												 // is not stored in the necessary way for MLPACK Convolutions.

			for (unsigned int k = 0; k < 3; ++k)
			{
				auto in = h5::read<std::vector<int> >(fd, "test_set_x", h5::offset{ i,0,0,k }, h5::count{ 1, 64, 64, 1 }, h5::stride{ testSetX.n_cols,1,1,3 });
				tempCol.insert(std::end(tempCol), std::begin(in), std::end(in));
			}
			testSetX.col(i) = arma::conv_to<arma::colvec>::from(tempCol);
		}
		testSetY = arma::mat(h5::read<arma::rowvec>(fd, "test_set_y"));
		classes = arma::mat(h5::read<arma::rowvec>(fd, "list_classes"));
	}

	// Normalize
	trainSetX /= 255;
	testSetX /= 255;

	// The model architecture as per the course example is:
	// (1)Conv2D->(2)ReLU->(3)MaxPool->(4)Conv2D->(5)ReLU->(6)MaxPool->(7)Flatten->(8)FullyConnected
	//
	// Since the course used 4x4 filters and 2x2 filters which are non-standard, I have changed these
	// to 5x5 and 3x3 filters. This makes the arithmeic easier for the padding calculation.
	// 
	//     Input Dim    Description
	//     ---------    -----------
	// (1)  64x64x3   -- Apply 8 filters of size 4x4 with stride = 2 and "Same" padding (= 2)
	// (2)  65x65x8   -- Standard ReLU layer
	// (3)  65x65x8   -- Pooling layer of size 8x8 with stride = 8
	// (4)  8x8x8     -- Apply 16 filters of size 2x2 with stride = 1 and "Same" padding (= 1)
	// (5)  9x9x16    -- Standard ReLU layer
	// (6)  9x9x16    -- Pooling layer of size 4x4 with stride = 4
	// (7)  2x2x16    -- Fully connected layer -> output = 6.
	// (10) 64x6      -- LogSoftMax layer

	FFN <NegativeLogLikelihood<>, XavierInitialization> model;
	model.Add<Convolution<> >(3, 8, 5, 5, 1, 1, 2, 2, 64, 64);
	model.Add<LeakyReLU<> >();
	model.Add<MaxPooling<> >(8, 8, 8, 8, true);
	model.Add<Convolution<> >(8, 16, 3, 3, 1, 1, 1, 1, 8, 8);
	model.Add<LeakyReLU<> >();
	model.Add<MaxPooling<> >(4, 4, 4, 4, true);
	model.Add<Linear<> >(64, 6);
	model.Add<LogSoftMax<> >();

	// Define the optimizer. Here we are using the Stochastic Gradient Descent with the Adam update function.
	// Arguments:
	// - learning rate = 0.009
	// - batchsize     = 64
	// - maxIterations = 10000
	AdamUpdate adamUpdate(1e-8, 0.9, 0.999);
	SGD<AdamUpdate> optimizer(0.009, 1, 1080, 1e-05, true, adamUpdate);

	trainSetY += 1;  // NegativeLogLikelihood expects output in range [1, N]. Data was in range [0,N-1] previously.
	testSetY += 1;   // NegativeLogLikelihood expects output in range [1, N]. Data was in range [0,N-1] previously.

	for (unsigned int epoch = 0; epoch < 100; ++epoch)
	{
		model.Train(trainSetX, trainSetY, optimizer);
		optimizer.ResetPolicy() = false;   // Keep the trained parameter settings from the previous epoch.
		std::cout << "\n\n\nEpoch #" << epoch << "\n\n";

		// Matrix to store the predictions on train and validation datasets.
		arma::mat     yPredict;
		arma::urowvec yPredictLabels;

		// Get predictions and accuracy on training data points.
		model.Predict(trainSetX, yPredict);
		yPredictLabels = arma::index_max(yPredict, 0) + 1;
		double trainAccuracy = arma::accu(yPredictLabels == trainSetY)*100.0 / (double)trainSetX.n_cols;

		// Get predictions and accuracy on test data points.
		yPredict.clear();
		model.Predict(testSetX, yPredict);
		yPredictLabels = arma::index_max(yPredict, 0) + 1;
		double testAccuracy = arma::accu(yPredictLabels == arma::round(testSetY))*100.0 / (double)testSetX.n_cols;

		std::cout << "Training Accuracy   = " << trainAccuracy << "%\n"
			<< "Test Accuracy = " << testAccuracy << "%\n\n";
	}

	data::Save("data/CNNSignsRGBModel.bin", "model", model, false);

	getchar();
}


