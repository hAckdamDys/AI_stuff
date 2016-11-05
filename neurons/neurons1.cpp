//https://www.youtube.com/watch?v=KkwX7FkLfug

#include "classNeurons.h"
#include "classNets.h"

int main(){
	srand(time(NULL));
	vector<unsigned> topology;
	const unsigned inputNeurons=2;
	const unsigned outputNeurons=1;
	topology.push_back(inputNeurons);//input
	topology.push_back(5);//hidden
	topology.push_back(5);//hidden
	topology.push_back(5);//hidden
	topology.push_back(outputNeurons);//output
	Net myNet(topology);
	
	vector<double> inputVals;
	
	vector<double> targetVals;
	
	vector<double> resultVals;
	
	char ch;//do czyszczenia po cin
	unsigned licznik=0;
	unsigned coIle=1;
	string inputStr;
	double tmpDoub;
	
	fstream stream1;
	stream1.open("wagi.nnw",ios::app);
	stream1<<endl<<endl;
	stream1<<"@@@((tanh(inputVals[0])-sqrt(abs(inputVals[1])))/2)@@@\n topology: 2-3-3-1,tanh(x),eta=0.15,alpha=0.5\nWAGI:\n";
	
	fstream stream2;
	stream2.open("wagi2.nnw",ios::in);
	
	getline(cin,inputStr);
	if(inputStr=="skip" || inputStr=="exit"){
		goto skiptrain;
	}
	while(1){
		++licznik;
		
		inputVals.clear();
		for(int i=0;i<inputNeurons;i++){
			inputVals.push_back((rand()/double(RAND_MAX))*2-1);
		}
		myNet.feedForward(inputVals);
		
		targetVals.clear();
		targetVals.push_back(inputVals[0]*inputVals[1]);
		myNet.backProp(targetVals);
		
		if(licznik%coIle==0){
			cout<<"Próba Nr. "<<licznik<<endl;
			
			
			cout<<"Inputs:\n";
			for(int i=0;i<inputVals.size();i++){
				cout<<inputVals[i]<<endl;
			}
			cout<<"Outputs:\n";
			for(int i=0;i<targetVals.size();i++){
				cout<<targetVals[i]<<endl;
			}
			cout<<"Net results:\n";
			myNet.getResults(resultVals);
			getline(cin,inputStr);
			if(inputStr=="save and show" || inputStr=="show and save"){
				stream1<<"Wagi po "<<licznik<<"próbach"<<endl;
				myNet.showWeights();
				myNet.saveWeights(stream1);
				getline(cin,inputStr);
			}
			else if(inputStr=="save"){
				stream1<<"Wagi po "<<licznik<<"próbach"<<endl;
				myNet.saveWeights(stream1);
				getline(cin,inputStr);
			}
			else if(inputStr=="show"){
				cout<<"MYNET1:\n";
				myNet.showWeights();
				getline(cin,inputStr);
			}
			else if(inputStr=="load"){
				myNet.loadWeights(stream2);
				getline(cin,inputStr);
			}
			else if(inputStr=="test"){
				++licznik;
				inputVals.clear();
				for(int i=0;i<inputNeurons;i++){
					cin>>tmpDoub;
					inputVals.push_back(tmpDoub);
				}
				myNet.feedForward(inputVals);
				
				targetVals.clear();
				targetVals.push_back(inputVals[0]*inputVals[1]);
				myNet.backProp(targetVals);
				inputStr="";
				cout<<"Próba Nr. "<<licznik<<endl;				
				cout<<"Inputs:\n";
				for(int i=0;i<inputVals.size();i++){
					cout<<inputVals[i]<<endl;
				}
				cout<<"Outputs:\n";
				for(int i=0;i<targetVals.size();i++){
					cout<<targetVals[i]<<endl;
				}
				cout<<"Net results:\n";
				myNet.getResults(resultVals);
				while ((ch = cin.get()) != '\n' && ch != EOF);//czyszczenie przed getlinem
				getline(cin,inputStr);
			}
			if(inputStr=="change"){
				cin>>coIle;
				while ((ch = cin.get()) != '\n' && ch != EOF);//czyszczenie przed getlinem
			}
			if(inputStr=="exit"){break;}
		}
	}
	goto theEnd;
	skiptrain:
	while(1){
		getline(cin,inputStr);
		if(inputStr=="save and show" || inputStr=="show and save"){
			stream1<<"Wagi po "<<licznik<<"próbach"<<endl;
			myNet.showWeights();
			myNet.saveWeights(stream1);
			getline(cin,inputStr);
		}
		else if(inputStr=="save"){
			stream1<<"Wagi po "<<licznik<<"próbach"<<endl;
			myNet.saveWeights(stream1);
			getline(cin,inputStr);
		}
		else if(inputStr=="show"){
			myNet.showWeights();
			getline(cin,inputStr);
		}		
		else if(inputStr=="load"){
			myNet.loadWeights(stream2);
			getline(cin,inputStr);
		}
		if(inputStr=="exit"){break;}
		++licznik;
		cout<<"Próba Nr. "<<licznik<<endl;
		
		cout<<"Inputs:\n";
		inputVals.clear();
		for(unsigned i=0;i<inputNeurons;++i){
			cin>>tmpDoub;
			inputVals.push_back(tmpDoub);
		}
		myNet.feedForward(inputVals);
		
		while ((ch = cin.get()) != '\n' && ch != EOF);//czyszczenie przed getlinem
		
		cout<<"Net results:\n";
		myNet.getResults(resultVals);
	}
theEnd:
	stream1<<"\nKONIEC\n";
	stream1.close();
	stream2.close();
	return 0;
}