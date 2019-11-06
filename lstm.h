#include <stdio.h>
#include <math.h>
#include <iostream>

double
sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double
tanh(double x)
{
	double a = exp(x);
	double b = exp(-x);

	return (a - b)/(a + b);
}

void ff(double x3[], double state[], double wih[], double whh[], double bih[], double bhh[]){
    double r[4*64], i, f, g, o;
    for (int z = 0; z < 4*64; z++)  *(r + z) = 0.0;

    for (int k = 0; k< 256; k ++){
        for (int j = 0; j < 64; j ++){
            r[k] += wih[k * 64 + j] * x3[j] + whh[k * 64 + j] * (*(state + 64 + j)); 
        }
        r[k] += bih[k] + bhh[k];
    }
    //r = i /f /g /o

    for (int ii = 0; ii < 64; ii++){
        i = r[ii];
        f = r[ii+64];
        g = r[ii+64*2];
        o = r[ii+64*3];
        *(state + ii) = *(state + ii) * sigmoid(f) + sigmoid(i) * tanh(g);  
        *(state + 64 + ii) = sigmoid(o) * tanh(*(state + ii));
    }
}

double *lstm_per_k(double xx[], double wih[], double whh[], double bih[], double bhh[]){

    static double state [2*64];
    static double output [64];
    static double x3[64];

    for (int z = 0; z < 2*64; z++)  *(state + z) = 0.0;

    for (int k = 0; k< ksize; k++){
        for (int i = 0; i < 64; i++){
                x3[i] = xx[k + i*8];
            }
        ff(x3, (double *)state, (double *)wih, (double *)whh, (double *)bih, (double *)bhh);  // from pytorch txt.
    }
    for (int j=0; j < 64; j++)  output[j] = *(state + j + 64);
    return output;
}

double *lstm(double *x, double wih[], double whh[], double bih[], double bhh[]){  //top funct.

    static double xx[ksize*64];
    static double lstm_out [64*1024];
    double *pointer;
    int count = 0;
for (int k = 0; k<1024; k++){
    int n = 0;
    for (int j = 0; j<64; j++){
        for (int i = 0; i < 8;i++){
            xx[n] = *(x + i + k*8 + j*1024*8);
            n++; 
            }
        }

    pointer = lstm_per_k(xx, (double *)wih, (double *)whh, (double *)bih, (double *)bhh);

    for (int i = 0; i<64;i++){
        lstm_out[count] = *(pointer + i);
        count ++;
         }
    }
    return lstm_out;   // note the size is (1024,64)!
}
