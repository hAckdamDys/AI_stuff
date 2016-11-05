#include "classNeurons.h"

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
	for (unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
		m_outputWeights.back().deltaWeight = 0;
		
	}
	m_myIndex = myIndex;
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex, istream& stream){
	for (unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		stream>>m_outputWeights.back().weight;
		stream>>m_outputWeights.back().deltaWeight;
		
	}
	m_myIndex = myIndex;
}

double Neuron::eta=0.15;
double Neuron::alpha=0.5;

void Neuron::showWeights(const unsigned nextLayerSize) const{
	for(unsigned i=0;i<nextLayerSize-1;++i){//-1 bo bias
		cout<<"		w="<<m_outputWeights[i].weight<<endl;
		cout<<"		dw="<<m_outputWeights[i].deltaWeight<<endl;
	}
}

void Neuron::saveWeights(const unsigned nextLayerSize, ostream& stream) const{
	for(unsigned i=0;i<nextLayerSize-1;++i){//-1 bo bias
		stream<<m_outputWeights[i].weight<<endl;
		stream<<m_outputWeights[i].deltaWeight<<endl;
	}
}

void Neuron::loadWeights(const unsigned nextLayerSize, istream& stream){
	for(unsigned i=0;i<nextLayerSize-1;++i){//-1 bo bias
		stream>>m_outputWeights[i].weight;
		stream>>m_outputWeights[i].deltaWeight;
	}
}

void Neuron::updateInputWeights(Layer &prevLayer){
	//The weights to be updated are in the connection
	//container in the neurons in preceding layer
	for(unsigned n=0;n<prevLayer.size();++n){
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		
		double newDeltaWeight = 
			// Individual input, magnified the gradient and train rate:
			eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
			
		neuron.m_outputWeights[m_myIndex].deltaWeight=newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight+=newDeltaWeight;
	}
}

void Neuron::calcOutputGradients(double targetVal){
	double delta = targetVal - m_outputVal;
	m_gradient=delta*Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const{
	double sum = 0.0;
	//Sum our contributions of the errors at the nodes we feed
	for(unsigned n=0;n<nextLayer.size()-1;++n){
		sum+=m_outputWeights[n].weight*nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
	double dow = sumDOW(nextLayer);
	m_gradient=dow*Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x){
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
	return tanhDerivative(x);
}

void Neuron::feedForward(const Layer &prevLayer){
	double sum = 0.0;
	
	//Sum the previous layer outputs
	//include the bias node from previous
	
	for(unsigned n = 0; n<prevLayer.size();++n){
		sum+=prevLayer[n].getOutputVal()*prevLayer[n].m_outputWeights[m_myIndex].weight;//wagi do siebie z poprzednich
	}
	m_outputVal = Neuron::transferFunction(sum);
}
