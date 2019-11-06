#include <stdio.h>
#include <math.h>
#include <iostream>
#include "gen.h"
#include "gen_encoder.h"
#include "gen_io.h"
#include "gen_b0l0.h"
#include "gen_b0l1.h"
#include "gen_b0p.h"
#include "gen_b0f.h"

#include "gen_b1l0.h"
#include "gen_b1l1.h"
#include "gen_b1p.h"
#include "gen_b1f.h"
#include "gen_out.h"

#include "functions.h"
#include "lstm.h"
#include "MHP.h"


double *mean_(double x[]){
	static double sum [o_dx];
	for (int i = 0; i < o_dx; i++){
		for (int j = 0; j< o_dy; j ++){
			sum[i] += x[i + j * o_dx];
		}
		sum[i] = sum[i] / o_dy;
	}
	return sum;  // dim 1024,1
}


double *std_(double x[]){

	double *me;  //[o_dx];
	double *me2; //[o_dx];
	
	me = mean_(x);

	for (int i = 0; i < o_dx ;i ++){ //1024
		for (int j =0; j< o_dy ;j++){  //64
				x[i + j * o_dx] -= *(me + i);
				x[i + j * o_dx] = x[i + j * o_dx] * x[i + j * o_dx];
		}
	}
	me2 = mean_(x);

	for (int i = 0; i < o_dx ; i ++){
		*(me2 + i) = sqrt(*(me2 + i));
	}
	return me2;
}


double *norm_(double *xx, double a[], double b[]){
	// input x : 1-D, 1024*64
	double *mean_p;
	double *std_p;
	static double out[o_dx*o_dy];
    static double x[o_dx*o_dy];
    
	static double mean_x [o_dx];
	static double std_x [o_dx];

    for (int i = 0; i < o_dx*o_dy; i ++){
        x[i] = *(xx + i);
	}

    mean_p = mean_(x);
	for (int i = 0; i < o_dx; i ++)   mean_x[i] = *(mean_p + i);
	std_p = std_(x);
	for (int i = 0; i < o_dx; i ++)   std_x[i] = *(std_p + i);
	
	int n = 0;
	for (int i = 0; i < o_dy; i ++){ //64
		for (int j = 0; j < o_dx; j ++){  //1024
			 out[n] = a[i] * (*(xx + i * o_dx + j) - mean_x[j]) / (std_x[j] + 1e-6) + b[i];
		    n++;	
		}
	}
	return out;
}


int main()
{
	static double *y;
    
	y = linear((double *) input, (double *) ew, eb, o_dy, o_dx, i_dy);

	static double y2 [o_dx * o_dy];
	for (int k = 0; k < o_dy * o_dx; k++){
		y2[k] = *(y + k);
	}

    ///////////////////////////////////////BLOCK_0 layer 0 (norm + lstm)
	y = norm_(y, b_0_l_0_na, b_0_l_0_nb);
	y = lstm_in(y);    // y.shape 1024 * 8 * 64
	y = lstm(y, (double *)b_0_l_0_wih, (double *)b_0_l_0_whh, (double *)b_0_l_0_bih, (double *)b_0_l_0_bhh);      //y shape 1024,64

	transpose(y,o_dx, o_dy);
	for (int k = 0; k < o_dy * o_dx; k++){
		*(y + k) += y2[k];
		y2[k] = *(y + k);
	}
	// block 0 layer 1
	y = norm_(y, b_0_l_1_na, b_0_l_1_nb);
	y = lstm_in(y);
	y = lstm(y, (double *)b_0_l_1_wih, (double *)b_0_l_1_whh, (double *)b_0_l_1_bih, (double *)b_0_l_1_bhh);      //y shape 1024,64

	transpose(y, o_dx, o_dy);
	for (int k = 0; k < o_dy * o_dx; k++){
		*(y + k) += y2[k];
		y2[k] = *(y + k);
	}
    // y2 is keept as original   y y2 shape  64,1024
	// MHP start
	y = norm_(y, b_0_c_0_na, b_0_c_0_nb);
	// prepare for q,v,k
	attention(y, (double *)b_0_p_w0, (double *)b_0_p_b0, (double *)b_0_p_w1, (double *)b_0_p_b1, (double *)b_0_p_w2, (double *)b_0_p_b2);
    // transpose&reshape
	y = out_of_attn(y, (double *)b_0_p_w3, (double *)b_0_p_b3);  // shape 64*1024

	for (int k = 0; k < o_dy * o_dx; k++){
		*(y + k) += y2[k];
		y2[k] = *(y + k);
	}

	y = norm_(y, b_0_c_1_na, b_0_c_1_nb);
	transpose(y, 64, 1024);
	y = linear1(y, (double *)b_0_f_w1, (double *)b_0_f_b1, 256, 1024, 64);
	// relu
	for (int r = 0; r<256*1024; r++)  *(y + r) = (*(y + r) < 0.0) ? 0.0 : *(y + r);
	transpose(y, 256, 1024);
	y = linear(y, (double *)b_0_f_w2, (double *)b_0_f_b2, 64, 1024, 256);
	for (int k = 0; k < o_dy * o_dx; k++){
		*(y + k) += y2[k];
		y2[k] = *(y + k);
	}

/////////////////////////////////////////BLOCK_1
	// block 1 layer 0 (norm + lstm)
	y = norm_(y, b_1_l_0_na, b_1_l_0_nb);
	y = lstm_in(y);    // y.shape 1024 * 8 * 64
	y = lstm(y, (double *)b_1_l_0_wih, (double *)b_1_l_0_whh, (double *)b_1_l_0_bih, (double *)b_1_l_0_bhh);      //y shape 1024,64

	transpose(y,o_dx, o_dy);
	for (int k = 0; k < o_dy * o_dx; k++){
		*(y + k) += y2[k];
		y2[k] = *(y + k);
	}
	// block 1 layer 1
	y = norm_(y, b_1_l_1_na, b_1_l_1_nb);
	y = lstm_in(y);
	y = lstm(y, (double *)b_1_l_1_wih, (double *)b_1_l_1_whh, (double *)b_1_l_1_bih, (double *)b_1_l_1_bhh);      //y shape 1024,64

	transpose(y, o_dx, o_dy);
	for (int k = 0; k < o_dy * o_dx; k++){
		*(y + k) += y2[k];
		y2[k] = *(y + k);
	}
    // y2 is keept as original   y y2 shape  64,1024
	// MHP start
	y = norm_(y, b_1_c_0_na, b_1_c_0_nb);
	// prepare for q,v,k
	attention(y, (double *)b_1_p_w0, (double *)b_1_p_b0, (double *)b_1_p_w1, (double *)b_1_p_b1, (double *)b_1_p_w2, (double *)b_1_p_b2);
    // transpose&reshape
	y = out_of_attn(y, (double *)b_1_p_w3, (double *)b_1_p_b3);  // shape 64*1024

	for (int k = 0; k < o_dy * o_dx; k++){
		*(y + k) += y2[k];
		y2[k] = *(y + k);
	}

	y = norm_(y, b_1_c_1_na, b_1_c_1_nb);
	transpose(y, 64, 1024);
	y = linear1(y, (double *)b_1_f_w1, (double *)b_1_f_b1, 256, 1024, 64);
	// relu
	for (int r = 0; r<256*1024; r++)  *(y + r) = (*(y + r) < 0.0) ? 0.0 : *(y + r);
	transpose(y, 256, 1024);
	y = linear(y, (double *)b_1_f_w2, (double *)b_1_f_b2, 64, 1024, 256);

/////////////////output layer
	static double output [24];
	for (int i = 0; i<64; i++){
		*(y + i) = *(y + 1023 + 1024 * i);
	}

	for (int j = 0; j<24 ; j++){
		for (int k = 0; k<64; k++)
			output[j] += lw[j][k] * (*(y + k));
		output[j] += lb[j];
	}
////////////////////print for check.
/*
	for (int i = 0; i < o_dx; i++){
		for (int j = 0; j < o_dy; j++){  //64
			printf("%.4f //", *(y + i + j * o_dx));
		}
		printf("\n");
	} */
	for (int i = 0; i<24 ; i++)   printf("%.4f//", output[i]);
	return 0;
}
