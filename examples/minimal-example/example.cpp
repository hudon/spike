#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <vector>
#include <iostream>

double dt = 0.001;       // simulation time step  
double t_rc = 0.02;      // membrane RC time constant
double t_ref = 0.002;    // refractory period
double t_pstc = 0.1;     // post-synaptic time constant
unsigned int N_samples = 100;  // number of sample points to use when finding decoders

// the input to the system over time
double input(double t){
	return sin(t);
}

// the function to compute between A and B
double function(double x){
	return x*x;
}

double idempotent_function(double x){
	return x;
}


void generate_gain_and_bias(unsigned int count, unsigned int intercept_low, unsigned int intercept_high, unsigned int rate_low, unsigned int rate_high, double * gain, double * bias){
	for(unsigned int i = 0; i < count; i++){
		// desired intercept (x value for which the neuron starts firing
		double intercept = (double)(intercept_high - intercept_low) / (double)RAND_MAX;
		// desired maximum rate (firing rate when x is maximum)
		double rate = (double)(rate_high - rate_low) / (double)RAND_MAX;

		// this algorithm is specific to LIF neurons, but should
		//  generate gain and bias values to produce the desired
		//  intercept and rate
		double z = 1.0 / (1-exp((t_ref-(1.0/rate))/t_rc));
		double g = (1 - z)/(intercept - 1.0);
		double b = 1 - g*intercept;
		gain[i] = g;
		bias[i] = b;
	}
}


//  voltage and v=1 is the firing threshold
void run_neurons(double * input, double *v, double * ref, double * spikes, unsigned int len){
	for(unsigned int i = 0; i < len; i++){
		double dV = dt * (input[i]-v[i]) / t_rc;    // the LIF voltage change equation
		v[i] += dV;
		if (v[i]<0)
			v[i]=0;                   // don't allow voltage to go below 0

		if (ref[i]>0 ){                       // if we are in our refractory period
			v[i]=0;                          //   keep voltage at zero and
			ref[i]-=dt;                     //   decrease the refractory period
}

		if ( v[i]>1 ){                         // if we have hit threshold
			spikes[i] = 1;            //   spike
			v[i] = 0;                        //   reset the voltage
			ref[i] = t_ref;                  //   and set the refractory period
		} else{
			spikes[i] = 0;
		}
	}
}

      
// measure the spike rate of a whole population for a given represented value x        
void compute_response(double x, double *encoder, double *gain, double *bias,  unsigned int len, double *count, double time_limit=0.5){
	unsigned int N = len;   // number of neurons
	double v[N];          // voltage
	double ref[N];        // refractory period
	memset(v,0,sizeof(double)*N);
	memset(ref,0,sizeof(double)*N);
    
	double input[N];
	for (unsigned int i = 0; i < N; i++){
		input[i] = x*encoder[i]*gain[i]+bias[i];
		v[i]= (double)(1) / (double)RAND_MAX;  // randomize the initial voltage level
	}
    
	memset(count,0,sizeof(double)*N);
	// feed the input into the population for a given amount of time
	double t = 0;
	while (t < time_limit){
		double spikes[N];
		run_neurons(input, v, ref, spikes, N);
		for (unsigned int i = 0; i < N; i++){
			if (spikes[i])
				count[i] += 1;
		}
		t += dt;
	}
	for(int i = 0; i < N; i++)
		count[i] /= time_limit;// return the spike rate (in Hz)
} 

    
// compute the tuning curves for a population    
void compute_tuning_curves(double * encoder, double *gain, double * bias, double * x_values, double * A){
	// generate a set of x values to sample at
	for(unsigned int i = 0; i < N_samples; i++)
		x_values[i] = i*2.0/N_samples - 1.0;

	// build up a matrix of neural responses to each input (i.e. tuning curves)
	for (unsigned int i = 0; i < N_samples; i++)
		compute_response(x_values[i], encoder, gain, bias, N_samples, &(A[N_samples * i]));
}

void compute_decoder(double * encoder, double * gain, double * bias, double (*f)(double)){
	// get the tuning curves
	double x_values[N_samples];
	double array_1[N_samples];
	double A[N_samples];
	compute_tuning_curves(encoder, gain, bias, x_values, A);

	// get the desired decoded value for each sample point
	for(unsigned int i = 0; i < N_samples; i++)
		array_1[i] = f(x_values[i]);

	//  TODO this function would requires c/c++ libraries
	//  Most notably the linalg.pinv which calculates the (Moore-Penrose) pseudo-inverse of a matrix.

	//  find the optimum linear decoder
	//  A=numpy.array(A).T
	//  Gamma=numpy.dot(A, A.T)
	//  Upsilon=numpy.dot(A, array_1)
	//  Ginv=numpy.linalg.pinv(Gamma)        
	//  decoder=numpy.dot(Ginv,Upsilon)/dt
	//  Not implemented
	//  return decoder
}


int main(int argc, char * argv[]){

	std::cout << "A lot of stuff is not implemented, so expect this to segfault.\n";
	std::cout.flush();

	srand ( time(NULL) );


	unsigned int N_A = 50;         // number of neurons in first population
	unsigned int N_B = 40;         // number of neurons in second population
	unsigned int rate_A[2];  // range of maximum firing rates for population A
	rate_A[0] = 25;
	rate_A[1] = 75;

	unsigned int rate_B[2]; // range of maximum firing rates for population B
	rate_B[0] = 50;
	rate_B[1] = 100;

	double encoder_A[N_A];
	double encoder_B[N_B];

	double gain_A[N_A];
	double bias_A[N_A];
	double gain_B[N_B];
	double bias_B[N_B];

////////////////////////////////////////
//// Step 1: Initialization
////////////////////////////////////////

	// create random encoders for the two populations
	for(unsigned int i = 0; i < N_A; i++)
		encoder_A[i] = rand() % 2 == 0 ? 1 : -1;

	for(unsigned int i = 0; i < N_B; i++)
		encoder_B[i] = rand() % 2 == 0 ? 1 : -1;
	

	// random gain and bias for the two populations
	generate_gain_and_bias(N_A, -1, 1, rate_A[0], rate_A[1],gain_A,bias_A);
	generate_gain_and_bias(N_B, -1, 1, rate_B[0], rate_B[1],gain_B,bias_B);

	//  TODO: actually compute the decoders
	double decoder_A[1][1];
	double decoder_B[1][1];
	//  find the decoders for A and B
	compute_decoder(encoder_A, gain_A, bias_A, &function);
	compute_decoder(encoder_B, gain_B, bias_B, &idempotent_function);

	//  compute the weight matrix
	//  TODO: actually calculate weights
	double weights[1][1];
	//weights=numpy.dot(decoder_A, [encoder_B])

	
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// Step 2: Running the simulation
	//////////////////////////////////////////////////////////////////////////////////////////////////

	double v_A[N_A];       // voltage for population A
	double ref_A[N_A];     // refractory period for population A
	double input_A[N_A];   // input for population A
	memset(v_A,0,sizeof(double)*N_A);
	memset(ref_A,0,sizeof(double)*N_A);
	memset(input_A,0,sizeof(double)*N_A);

	double v_B[N_B];       // voltage for population B     
	double ref_B[N_B];     // refractory period for population B
	double input_B[N_B];   // input for population B
	memset(v_B,0,sizeof(double)*N_B);
	memset(ref_B,0,sizeof(double)*N_B);
	memset(input_B,0,sizeof(double)*N_B);

	// scaling factor for the post-synaptic filter
	double pstc_scale = 1.0-exp(-dt/t_pstc);


	// for storing simulation data to plot afterward
	std::vector<double> inputs;
	std::vector<double> times;
	std::vector<double> outputs;
	std::vector<double> ideal;

	double output=0.0;            // the decoded output value from population B
	double t = 0;

	while(t<10.0){
		// call the input function to determine the input value
		double x=input(t);

		// convert the input value into an input for each neuron
		for(unsigned int i = 0; i < N_A; i++)
			input_A[i] = x*encoder_A[i] * gain_A[i] + bias_A[i];

		// run population A and determine which neurons spike
		double spikes_A[N_A];
		run_neurons(input_A, v_A, ref_A, spikes_A, N_A);

		// decay all of the inputs (implementing the post-synaptic filter)            
		for(unsigned int j = 0; j < N_B; j++)
			input_B[j] *= (1.0-pstc_scale);

		// for each neuron that spikes, increase the input current
		//  of all the neurons it is connected to by the synaptic
		//  connection weight
		for(unsigned int i = 0; i < N_A; i++){
			if (spikes_A[i]){
				for(unsigned int j = 0; j < N_A; j++){
					input_B[j] += weights[i][j]*pstc_scale;
				}
			}
		}

		// compute the total input into each neuron in population B
		//  (taking into account gain and bias)    
		double total_B[N_B];
		memset(total_B,0,sizeof(double)*N_B);
		for (unsigned int j = 0; j < N_B; j++)
			total_B[j]=gain_B[j]*input_B[j]+bias_B[j];

		// run population B and determine which neurons spike
		double spikes_B[N_B];
		run_neurons(input_B, v_B, ref_B, spikes_B, N_B);

		// for each neuron in B that spikes, update our decoded value
		//  (also applying the same post-synaptic filter)
		output *= (1.0-pstc_scale);
		for(unsigned int i = 0; i < N_B; i++){
			if (spikes_B[i]){
				output += decoder_B[i][0]*pstc_scale;
			}
		}

		std::cout << t << output;
		times.push_back(t);
		inputs.push_back(x);
		outputs.push_back(output);
		ideal.push_back(function(x));
		t+=dt;
	}
}


