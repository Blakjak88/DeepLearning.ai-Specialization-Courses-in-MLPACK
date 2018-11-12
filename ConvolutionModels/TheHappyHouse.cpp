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
	// # Requires HDF5 and H5CPP libraries. (Note, need to run v10+ of HDF5
	// # which currently requires downloading the source and compiling.
	//
	// # Note:   The Python model in the exercise results in a training
	// #         accuracy of 94.1% and a test accuracy of 78.3%. This
	// #         MLPACK model should obtain 97.8% train accuracy and 92%
	// #         test accuracy.
	//
	// # Inputs: Four inputs required on the commnadline. W, X, Y, Z, where
	// #         W = learningRate, X = batchSize, Y = maxiterations,
	// #         Z = nbr of epochs.
    // #
    // # Suggested inputs:
	// #        LearningRate = 0.001
	// #        BatchSize = 16
	// #        MaxIterations = 600  (i.e. size of training dataset)
	// #        Epochs = 40
	//
	//
	// # Author:  David Armour
	//
	////////////////////////////////////////////////////////////////////////

	if (argc != 5)
	{
      std::cout << "\nWrong number of parameters. Correct usage is:\n\n\tTheHappyHouse W X Y Z\n\n\twhere,\tW = learning rate\n\t\tX = batchSize\n\t\tY = maxIterations\n\t\tZ = nbr of epochs.\n" << std::endl;
      return -1;
	}

   static const unsigned int TRAIN_SAMPLE_QTY = 600;
   static const unsigned int TEST_SAMPLE_QTY = 150;

	arma::mat trainSetX(64 * 64 * 3, TRAIN_SAMPLE_QTY);
	arma::mat trainSetY(1, TRAIN_SAMPLE_QTY);
	arma::mat testSetX(64 * 64 * 3, TEST_SAMPLE_QTY);
	arma::mat testSetY(1, TEST_SAMPLE_QTY);
	arma::mat trainX, testX, trainY, testY;
	arma::mat classes;

	// Load the Signs Data from the .h5 files. The data needs to be loaded in (64x64 + 64x64 + 64x64) x m  (m = nbr of samples).
	{
		auto fd = h5::open("data/train_happy.h5", H5F_ACC_RDONLY);
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
		auto fd = h5::open("data/test_happy.h5", H5F_ACC_RDONLY);
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
	// (1)Conv2D->(2)BatchNorm->(3)ReLU->(4)MaxPool->(5)(X)(Flatten->)(5)FullyConnected->(6)Sigmoid
	//
	//     Input Dim     Output Dim     Description
	//     ---------     ----------     -----------
	// (1)  64x64x3      64x64x32       -- Apply 16 filters of size 3x3 with stride = 1 and "Same" padding (= 1)
	// (2)  64x64x32     64x64x32       -- Apply a Batch normalisation layer after the Convolution.
	// (3)  64x64x32     64x64x32       -- LeakyReLU layer
	// (4)  64x64x32     32x32x32       -- Pooling layer of size 2x2 with stride = 2
	// (X)  32x32x32     32768x1        -- Flatten (this is taken care of by the Linear layer input in MLPACK)
	// (5)  32768x1      1x1            -- Fully connected layer -> output = 1.
	// (6)  1x1          1x1            -- Sigmoid layer

	FFN <CrossEntropyError<>, XavierInitialization> model;
	model.Add<Convolution<> >(
		3,		// Nbr of input layers. (e.g. an RGB image of 64x64x3 has 3 layers, where a grayscale image of 64x64x1 has 1 layer.
		32,	// Nbr of output layers/maps.
		7,		// Filter Width
		7,		// Filter Height
		1,		// Stride for Width
		1,		// Stride for Height
		3,		// Padding for Width
		3,		// Padding for Heigth
		64,   // Input Height
		64);	// Output Height
	model.Add<BatchNorm<> >(131072);   // 64x64x32 inputs to normalise
	model.Add<LeakyReLU<> >();
	model.Add<MaxPooling<> >(
		2,		// Width of Pooling filter
		2,		// Height of Pooling filter
		2,		// Stride for Width
		2,		// Stride for Height
		true);// True = floor / False = ceil (used for output dimension formula to round up or down if non-integer result)
	model.Add<Linear<> >(32768, 1);
	model.Add<SigmoidLayer<> >();

	// Define the optimizer. Here we are using the Stochastic Gradient Descent with the Adam update function.
	// Arguments:
	// - learning rate = 0.001
	// - batchsize     = 1st input
	// - maxIterations = 2nd input    (Note: maxIterations is incremented by batchSize at each loop thru' a batch so this
	//                                       needs to be set to the number of examples if you want one pass thru' the dataset.
	AdamUpdate adamUpdate(1e-8, 0.9, 0.999);
	SGD<AdamUpdate> optimizer(std::stod(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), 1e-05, true, adamUpdate);

	for (int epoch = 0; epoch < std::stoi(argv[4]); ++epoch)
	{
		model.Train(trainSetX, trainSetY, optimizer);
		optimizer.ResetPolicy() = false;   // Keep the initial and trained parameter settings from the previous epoch.

		// Matrix to store the predictions on train and validation datasets.
		arma::mat yPredict;

		// Get predictions and accuracy on training data points.
		model.Predict(trainSetX, yPredict);
		yPredict = arma::conv_to<arma::mat>::from(yPredict > 0.5);
		double trainAccuracy = arma::accu(yPredict == trainSetY)*100.0 / (double)trainSetX.n_cols;

		// Get predictions and accuracy on test data points.
		model.Predict(testSetX, yPredict);
		yPredict = arma::conv_to<arma::mat>::from(yPredict > 0.5);
		double testAccuracy = arma::accu(yPredict == testSetY)*100.0 / (double)testSetX.n_cols;

		if (!((epoch+1)%5))
		   std::cout << "Epoch: " << epoch+1 << "\t" << "Training Accuracy   = " << trainAccuracy << "%\tTest Accuracy = " << testAccuracy << "%" << std::endl;
	}

	data::Save("data/CNNHappyHouseModel.bin", "model", model, false);   // Save the trained model.

	return 0;
}
