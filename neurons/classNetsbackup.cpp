#include "classNeurons.h"
#include "classNets.h"
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
Net::Net(const vector<unsigned> &topology, istream& stream){
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		unsigned numOutputs = (layerNum == (topology.size() - 1)) ? 0 : topology[layerNum + 1];//bo neurony musza wiedziec ile maja pod sobą neuronów a ostatnie mają 0
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum];++neuronNum){//<= bo extra bias neuron
			m_layers.back().push_back(Neuron(numOutputs, neuronNum, stream));
			cout<<"Made a "<<neuronNum<<"			Neuron!\n";
		}
		//bias musi miec output 1
		m_layers.back().back().setOutputVal(1.0);
	}
	stream.seekg(0);
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
	//{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}
	//{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}
	//OGARNĄĆ TO!!!
	//{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}
	//{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}{!}
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
		cout<<resultVals.back()<<endl;
	}
	cout<<"m_error= "<<m_error<<endl;
}

void Net::showWeights() const{
	for(unsigned i=0;i<m_layers.size()-1;++i){
		cout<<"Layer "<<i<<endl;
		for(unsigned j=0;j<m_layers[i].size();++j){
			cout<<"	Neuron "<<j<<endl;
			m_layers[i][j].showWeights(m_layers[i+1].size());
		}
	}
}

void Net::saveWeights(ostream& stream) const{
	for(unsigned i=0;i<m_layers.size()-1;++i){
		//cout<<"Layer "<<i<<endl;
		for(unsigned j=0;j<m_layers[i].size();++j){
			//cout<<"	Neuron "<<j<<endl;
			m_layers[i][j].saveWeights(m_layers[i+1].size(),stream);
		}
	}
	stream<<endl;
}

void Net::loadWeights(istream& stream){
	for(unsigned i=0;i<m_layers.size()-1;++i){
		//cout<<"Layer "<<i<<endl;
		for(unsigned j=0;j<m_layers[i].size();++j){
			//cout<<"	Neuron "<<j<<endl;
			m_layers[i][j].loadWeights(m_layers[i+1].size(),stream);
		}
	}
	stream.seekg(0);
}
