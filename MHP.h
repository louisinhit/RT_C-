#include <stdio.h>
#include <math.h>
#include <iostream>

double *linear(double *x, double w[], double b[], const int ni, const int nj, const int nk)
{
    // encoder: weight 64 * 2,, input 1024 * 2,, bias 64.  output 64 * 1024
	// mhp: weight 64 * 64,, input 1024 * 64,, output 64 * 1024
    static double ly[o_dy * o_dx];
    for (int z = 0; z< ni * nj ; z++) ly[z] = 0.0;
    axi: for (int i = 0; i < ni; i++) //64
    {
		axj: for (int j = 0; j < nj; j++) //1024
		{
            	for (int k = 0; k < nk; k++){
				ly[i * nj + j] += w[i * nk + k] * (*(x + k + j * nk));
			}
		}
    }  // add bias  (y shape 64,1024)
	for (int i = 0; i < ni; i++){
		for (int j = 0; j < nj; j++){
			ly[i * nj + j] += b[i];
		}	
	}
	return ly;
}

double *linear1(double *x, double w[], double b[], const int ni, const int nj, const int nk)
{
	//  feed forward
    static double ly1[1024*256];
    for (int z = 0; z< ni * nj ; z++) ly1[z] = 0.0;
    axi: for (int i = 0; i < ni; i++) //64
    {
		axj: for (int j = 0; j < nj; j++) //1024
		{
            	for (int k = 0; k < nk; k++){
				ly1[i * nj + j] += w[i * nk + k] * (*(x + k + j * nk));
			}
		}
    }  // add bias  (y shape 64,1024)
	for (int i = 0; i < ni; i++){
		for (int j = 0; j < nj; j++){
			ly1[i * nj + j] += b[i];
		}	
	}
	return ly1;
}

void softmax(double *sc){
	int step = 2 * 1024;
	double sum[step];
	// need to zero sum after each call
	for (int j = 0; j < step; j++) sum[j] = 0;

	for (int k = 0; k < step; k++){
		for (int i = 0; i < 1024; i++)
			sum[k] += exp(*(sc + i + k * 1024));
		for (int i = 0; i < 1024; i++)
			*(sc + i + k * 1024) = exp(*(sc + i + k * 1024)) / sum[k];
	}
}

void attention(double *x, double pw0[], double pb0[], double pw1[], double pb1[], double pw2[], double pb2[]){
	
	static double *y3;
	double *pointer;
	
	static double qu[o_dx*o_dy];
	static double va[o_dx*o_dy];
	static double ke[o_dx*o_dy];
	
	y3 = copy(x);
	transpose(y3, 64, 1024);   //new y3 shape 1024,64
	// get qu, ke, va
	// calculate query  2,1024,32
	pointer = linear(y3, pw0, pb0, o_dy, o_dx, o_dy);
	transpose(pointer, 64, 1024);
	int count = 0;
	for (int n =0; n <2; n++){
		for (int i = 0; i<o_dx ; i++){
			for (int j = 0; j<32; j++){
				qu[count] = *(pointer + j + i * o_dy + 32 * n);
				count ++;
			}
		}
	}
    // key  2,1024,32
	pointer = linear(y3, pw1, pb1, o_dy, o_dx, o_dy);
	transpose(pointer, 64, 1024);
	count = 0;
	for (int n =0; n <2; n++){
		for (int i = 0; i<o_dx ; i++){
			for (int j = 0; j<32; j++){
				ke[count] = *(pointer + j + i * o_dy + 32 * n);
				count ++;
			}
		}
	}
	// value  2,1024,32
	pointer = linear(y3, pw2, pb2, o_dy, o_dx, o_dy);
	transpose(pointer, 64, 1024);
	count = 0;
	for (int n =0; n <2; n++){
		for (int i = 0; i<o_dx ; i++){
			for (int j = 0; j<32; j++){
				va[count] = *(pointer + j + i * o_dy + 32 * n);
				count ++;
			}
		}
	}
	static double scores[2*1024*1024];
	int base_score = 1024 * 1024;
	int base_qk = 1024 * 32;

	for (int z=0; z<base_score*2; z++)  scores[z] = 0.0;
	for (int h= 0; h < 2; h ++){
		for (int i = 0; i<o_dx; i++){
			for (int j = 0; j<o_dx ; j++){
				for (int k = 0; k<32 ; k++){
					scores[i * o_dx + j + h * base_score] += qu[k + i * 32 + h * base_qk] * ke[k + j * 32 + h * base_qk];
				}
			}
		}
	}
	  for (int m=0; m<base_score*2; m++)
		scores[m] /= sqrt(32);

	softmax((double *)scores);
	for (int z = 0; z<o_dx*o_dy; z++) *(x + z) = 0.0;
	
	for (int h = 0; h < 2 ; h++)
		for (int k = 0; k< 1024; k++)
			for (int n = 0; n<32 ; n++)
				for (int i = 0; i < 1024 ; i++)
					*(x + n + k * 32 + h * base_qk) += *(scores + i + k * 1024 + h * base_score) * (*(va + i * 32 + n + h * base_qk));

    //for (int m=0; m<base_qk*2; m++)		printf("%.4f //", *(x+m));
}

double *out_of_attn(double *x, double pw3[], double pb3[]){  //traspose 2*1024*32 to 1024*64, then linear.
	double temp[o_dy*o_dx];
	int cnt = 0;
	for (int p = 0; p<2 ; p++)
		for (int j = 0; j<1024 ;j++)
			for (int i = 0; i<32 ; i++){
				temp[i + j * 64 + p * 32] = *(x + cnt);
				cnt ++;
			}
    for (int i = 0; i<o_dy*o_dx ; i++)   *(x + i) = temp[i];  // current x shape 1024*64
	x = linear(x, (double *)pw3, pb3, o_dy, o_dx, o_dy);
	return x;
}
