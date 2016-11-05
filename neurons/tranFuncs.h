inline double eRelu(double x){
	return log(exp(x)+1);
}
inline double eReluDerivative(double x){
	return exp(x)/(exp(x)+1);
}
inline double leakyRelu(double x){
	return x>0.0?x:0.1*x;
}
inline double leakyReluDerivative(double y){
	return y>0.0?1:0.1;
}
inline double tanhDerivative(double y){
	return 1.0 - y*y;
}
inline double sigmoid(double x){
	return 1/(1+exp(-x));
}
inline double sigmoidDerivative(double y){
	return (1-y)*y;
}