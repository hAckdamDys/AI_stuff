#ifndef classNets
#define classNets
class Net{
	public:
		Net(const vector<unsigned> &topology);
		Net(const vector<unsigned> &topology, istream& stream);//load wag included
		void feedForward(const vector<double> &inputVals);
		void backProp(const vector<double> &targetVals);
		void getResults(vector<double> &resultVals) const;
		void showWeights() const;
		void saveWeights(ostream& stream) const;
		void loadWeights(istream& stream);
	private:
		vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;
};
#endif