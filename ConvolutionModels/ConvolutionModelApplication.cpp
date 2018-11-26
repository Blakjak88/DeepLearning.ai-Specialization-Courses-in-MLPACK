#include <mlpack/core.hpp>
#include <armadillo>							// !!! Must go after mlpack/core.hpp   ref: email from Ryan Curtin. !!!
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
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

int main(int argc, char* argv[])
{
	////////////////////////////////////////////////////////////////////////
	//
	// Convolution Model - Application
	//
	// # Imports 1,080 training images and 120 testing images of dimension
	// # 64x64x3 from .H5 databases and runs a Conv net replicating the
	// # DeepLearning.ai exercise model.
	//
	// # Requires HDF5 and H5CPP libraries.
	//
	// # Note:  The Python model in the exercise results in a training
	// #        accuracy of 94.1% and a test accuracy of 78.3%. By slightly
	// #        changing the model (filter dims) and using the following
	// #        parameters the result with MLPACK is a training accuracy of
	// #        97.3% and a test accuracy of 81.7%.
	// #
	// #        LearningRate = 0.009
	// #        BatchSize = 64
	// #        MaxIterations = 1080  (i.e. size of training dataset)
	// #        Epochs = 50
	// #
	// #        This model has high variance (overfitting) so introducing 
	// #        regularisaton (i.e. DropoutLayer) would help improve the
	// #        test result.
	//
	// # Author:  David Armour (11 Nov 2018)
	//
	////////////////////////////////////////////////////////////////////////

	if (argc != 5)
	{
      std::cout << "\nWrong number of parameters. Correct usage is:\n\n\tConvolutionModelApplication W X Y Z\n\n\twhere,\tW = learning rate\n\t\tX = batchSize\n\t\tY = maxIterations\n\t\tZ = nbr of epochs.\n" << std::endl;
      return -1;
	}

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
	// (1)Conv2D->(2)ReLU->(3)MaxPool->(4)Conv2D->(5)ReLU->(6)MaxPool->(X)(Flatten->)(7)FullyConnected->(8)LogSoftMax
	//
	//     Input Dim    Description
	//     ---------    -----------
	// (1)  64x64x3   -- Apply 8 filters of size^ 5x5 with stride = 2 and "Same" padding (= 2)
	// (2)  65x65x8   -- Leaky ReLU layer (better than ReLU)
	// (3)  65x65x8   -- Pooling layer of size 8x8 with stride = 8
	// (4)  8x8x8     -- Apply 16 filters of size^ 3x3 with stride = 1 and "Same" padding (= 1)
	// (5)  9x9x16    -- Leaky ReLU layer (better than ReLU)
	// (6)  9x9x16    -- Pooling layer of size 4x4 with stride = 4
	// (X)  2x2x16    -- Flatten (this is taken care of by the Linear layer input in MLPACK)
	// (7)  64x6      -- Fully connected layer -> output = 6.
	// (8)  6x6       -- LogSoftMax layer
	//
	// Note:   ^  The Coursera course example uses 4x4 and 2x2 filters but I found that using 5x5 and 3x3
	//            converges much better. Also the course uses ReLU layers but LeakyReLU works better.

	FFN <NegativeLogLikelihood<>, XavierInitialization> model;
	model.Add<Convolution<> >(
		3,		// Nbr of input layers. (e.g. an RGB image of 64x64x3 has 3 layers, where a grayscale image of 64x64x1 has 1 layer.
		8,		// Nbr of output layers/maps.
		5,		// Filter Width
		5,		// Filter Height
		1,		// Stride for Width
		1,		// Stride for Height
		2,		// Padding for Width
		2,		// Padding for Heigth
		64,   // Input Height
		64);	// Output Height
	model.Add<LeakyReLU<> >();
	model.Add<MaxPooling<> >(
		8,		// Width of Pooling filter
		8,		// Height of Pooling filter
		8,		// Stride for Width
		8,		// Stride for Height
		true);// True = floor / False = ceil (used for output dimension formula to round up or down if non-integer result)
	model.Add<Convolution<> >(8, 16, 3, 3, 1, 1, 1, 1, 8, 8);
	model.Add<LeakyReLU<> >();
	model.Add<MaxPooling<> >(4, 4, 4, 4, true);
	model.Add<Linear<> >(64, 6);
	model.Add<LogSoftMax<> >();

	// Define the optimizer. Here we are using the Stochastic Gradient Descent with the Adam update function.
	// Arguments:
	// - learning rate = 1st input (e.g. 0.009)
	// - batchsize     = 2nd input (e.g. 64)
	// - maxIterations = 3rd input (e.g. 1080)    (Note: Nbr of iterations is incremented by batchSize at each loop thru' a batch so this
	//                                                   needs to be set to the number of examples if you want one pass thru' the dataset.
	AdamUpdate adamUpdate(1e-8, 0.9, 0.999);
	SGD<AdamUpdate> optimizer(std::stod(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), 1e-05, true, adamUpdate);

	trainSetY += 1;  // NegativeLogLikelihood expects output in range [1, N]. Data was in range [0,N-1] previously.

	for (int epoch = 0; epoch < std::stoi(argv[4]); ++epoch)
	{
		model.Train(trainSetX, trainSetY, optimizer);
		optimizer.ResetPolicy() = false;   // Keep the initial and trained parameter settings from the previous epoch.

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
		yPredictLabels = arma::index_max(yPredict, 0);
		double testAccuracy = arma::accu(yPredictLabels == testSetY)*100.0 / (double)testSetX.n_cols;

		if (!((epoch+1)%5))
		   std::cout << "Epoch: " << epoch+1 << "\t" << "Training Accuracy   = " << trainAccuracy << "%"	<< "\tTest Accuracy = " << testAccuracy << "%" << std::endl;
	}

	data::Save("data/CNNSignsRGBModel.bin", "model", model, false);   // Save the trained model.

	return 0;
}
