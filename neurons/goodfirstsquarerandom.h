//https://www.youtube.com/watch?v=KkwX7FkLfug

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

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
		void setOutputVal(double val){m_outputVal=val;}
		double getOutputVal(void) const {return m_outputVal;}
		void feedForward(const Layer &prevLayer);
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const Layer &nextLayer);
		void updateInputWeights(Layer &prevLayer);
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
};

double Neuron::eta=0.15;
double Neuron::alpha=0.5;

void Neuron::updateInputWeights(Layer &prevLayer){
	//The weights to be updated are in the connection
	//container in the neurons in preceding layer
	for(unsigned n=0;n<prevLayer.size();++n){
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		
		double newDeltaWeith = 
			// Individual input, magnified the gradient and train rate:
			eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
			
		neuron.m_outputWeights[m_myIndex].deltaWeight=newDeltaWeith;
		neuron.m_outputWeights[m_myIndex].weight+=newDeltaWeith;
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
	//tanh - output [-1,1]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
	//tanh - output [-1,1]
	return 1.0 - x*x;
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
	for (unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
		
	}
	m_myIndex = myIndex;
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

class Net{
	public:
		Net(const vector<unsigned> &topology);
		void feedForward(const vector<double> &inputVals);
		void backProp(const vector<double> &targetVals);
		void getResults(vector<double> &resultVals) const;
	private:
		vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;
};

Net::Net(const vector<unsigned> &topology){
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		unsigned numOutputs = (layerNum == (topology.size() - 1)) ? 0 : topology[layerNum + 1];//bo neurony musza wiedziec ile maja pod sobą neuronów a ostatnie mają 0
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum];++neuronNum){//<= bo extra bias neuron
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout<<"Made a "<<neuronNum<<"			Neuron!\n";
		}
		//bias musi miec output 1
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const vector<double> &inputVals){
	assert(inputVals.size() == m_layers[0].size()-1);//to musi byc prawda, -1 bo bias
	
	//Przydzielanie input do neuronów
	for(unsigned i = 0;i < inputVals.size();++i){
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	
	//Forward propagate:
	for(unsigned layerNum =1;layerNum < m_layers.size();++layerNum){
		Layer &prevLayer = m_layers[layerNum - 1];
		for(unsigned n=0;n<m_layers[layerNum].size()-1;++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const vector<double> &targetVals){

	Layer &outputLayer = m_layers.back();
	m_error = 0.0;
	//Calculate overall net error (RMS):
	//{
	//to jest funkcja po prostu
	for (unsigned n=0;n<outputLayer.size() - 1;++n){
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error +=delta * delta;
	}
	m_error /= outputLayer.size() - 1;//get average error squared
	m_error = sqrt(m_error);//rms
	//}
	
	//Implement a recent average measurment:
	//{
	m_recentAverageError=(m_recentAverageError*m_recentAverageSmoothingFactor+m_error)/(m_recentAverageSmoothingFactor + 1.0);
	
	//}
	
	//Gradients:
	//{
	//Calculate output layer gradients
	for(unsigned n = 0;n<outputLayer.size()-1;++n){//-1 bo bez bias output
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}
	//Calculate gradients on hidden layers
	for(unsigned layerNum = m_layers.size()-2;layerNum>0;--layerNum){//-2 bo -1 to jest ostatni output
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum+1];
		for(unsigned n = 0; n< hiddenLayer.size();++n){
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}
	//}

	//For all layers from outputs to first hidden layer
	//update connection weights
	//{
	for(unsigned layerNum = m_layers.size()-1;layerNum>0;--layerNum){
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum-1];
		
		for(unsigned n=0;n<layer.size()-1;++n){
			layer[n].updateInputWeights(prevLayer);
		}
		
	}
	
	
	//}
}

void Net::getResults(vector<double> &resultVals) const{
	
	resultVals.clear();
	
	for(unsigned n=0;n<m_layers.back().size()-1;++n){
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

int main(){
	srand(time(NULL));
	vector<unsigned> topology;
	topology.push_back(3);//input
	topology.push_back(2);//hidden
	topology.push_back(1);//output
	Net myNet(topology);
	
	vector<double> inputVals;
	
	vector<double> targetVals;
	
	vector<double> resultVals;
	
	unsigned licznik=0;
	double tmpDouble;
	while(1){
		++licznik;
		if(licznik%100000==0){
			cout<<"Próba Nr. "<<licznik<<endl;
		
			inputVals.clear();
			tmpDouble=rand()/double(RAND_MAX);
			cout<<"pierwszy input to: "<<tmpDouble;
			inputVals.push_back(tmpDouble);
			tmpDouble=rand()/double(RAND_MAX);
			cout<<" ,drugi input to: "<<tmpDouble;
			inputVals.push_back(tmpDouble);
			tmpDouble=rand()/double(RAND_MAX);
			cout<<" ,trzeci input to: "<<tmpDouble<<endl;
			inputVals.push_back(tmpDouble);
			myNet.feedForward(inputVals);
			
			targetVals.clear();
			tmpDouble=(inputVals[0]*inputVals[0]);//pierwszy kwadrat
			cout<<"pierwszy output to: "<<tmpDouble<<"\n Wyniki:\n";
			targetVals.push_back(tmpDouble);
			myNet.backProp(targetVals);
			
			myNet.getResults(resultVals);
			cout<<resultVals[0]<<endl;
			cin.get();
		
		}
		else{
			inputVals.clear();
			tmpDouble=rand()/double(RAND_MAX);
			inputVals.push_back(tmpDouble);
			tmpDouble=rand()/double(RAND_MAX);
			inputVals.push_back(tmpDouble);
			tmpDouble=rand()/double(RAND_MAX);
			inputVals.push_back(tmpDouble);
			myNet.feedForward(inputVals);
			
			targetVals.clear();
			tmpDouble=(inputVals[0]*inputVals[0]);//pierwszy kwadrat
			targetVals.push_back(tmpDouble);
			myNet.backProp(targetVals);
		}
	}
	return 0;
}