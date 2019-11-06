#include <stdio.h>
#include <math.h>
#include <iostream>
#define ksize 8
#define seq_len (1024+ksize-1)


int *idx_gen(){

    static int idx [1024*ksize];
    int n = 0;
    for (int j = ksize -1;j < seq_len; j++){
        for (int i = j-(ksize-1); i < j + 1;i++){
            idx[n] = i;
            n ++;
        }
    }
    return idx;
}

double *lstm_in(double *y){

    double yy[1024*64];   //(7+1024)*64
    double new_array[seq_len*64];
    double temp[seq_len*64];
    int *idx = idx_gen();
    int cc = 0;
    
    for (int ii = 0; ii<  1024; ii++){
        for(int kk = 0; kk< 64; kk++){
           yy[cc] = *(y + ii + kk * 1024);
           cc++;
        }
    }
    for (int m = 0; m<o_dx*o_dy ; m++){
        temp[7*64 + m] = yy[m];
    }

    for (int i = 0; i < seq_len; ++i ){
       for (int j = 0; j < o_dy; ++j ){
          int index1 = i*o_dy+j;
          int index2 = j*seq_len+i;
          new_array[index2] = temp[index1];
       }
    }
    static double new_y [1024*ksize*64];
    for (int i = 0; i < 64; i++){
        for (int n = 0; n < 1024*ksize; n++){
            new_y[n + i * 1024 * ksize] = new_array[*(idx + n) + i * seq_len];
        }
    }
    return new_y;
}

void transpose(double *in, int iny, int inx){
    
    double new_array[inx*iny];
    for (int i = 0; i < iny; ++i ){
       for (int j = 0; j < inx; ++j ){
          int index1 = i*inx+j;
          int index2 = j*iny+i;
          new_array[index2] = *(in + index1);
       }
    }
    for (int j = 0; j < inx*iny; j++){
        *(in + j) = new_array[j];
    }
}

double *copy(double *y){
    static double co[o_dx*o_dy];
    for (int i = 0; i<o_dx*o_dy; i++){
        co[i] = *(y + i);
    }
    return co;
}
