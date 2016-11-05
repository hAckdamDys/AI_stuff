#ifndef classNeurons
#define classNeurons
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <fstream>
#include "tranFuncs.h"
using namespace std;
struct Connection{
	double weight;
	double deltaWeight;
};
class Neuron;
typedef vector<Neuron> Layer;
class Neuron{
	public:
		Neuron(unsigned numOutputs, unsigned myIndex);
		Neuron(unsigned numOutputs, unsigned myIndex, istream& stream);
		void setOutputVal(double val){m_outputVal=val;}
		double getOutputVal(void) const {return m_outputVal;}
		void feedForward(const Layer &prevLayer);
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const Layer &nextLayer);
		void updateInputWeights(Layer &prevLayer);
		void showWeights(const unsigned nextLayerSize) const;
		void saveWeights(const unsigned nextLayerSize, ostream& stream) const;
		void loadWeights(const unsigned nextLayerSize, istream& stream);
	private:
		static double eta;		//[0.0 1.0] overall net training rate
		static double alpha;	//[0.0 n] multiplier of last weight change (momentum)
		static double transferFunction(double x);// to jest ta funkcja np. tanh, relu, sigmoid
		static double transferFunctionDerivative(double x);
		static double randomWeight(){return rand()/double(RAND_MAX);}
		double sumDOW(const Layer &nextLayer) const;
		double m_outputVal;
		vector<Connection> m_outputWeights;
		unsigned m_myIndex;
		double m_gradient;
		double m_sum;
};
#endif