#include <math.h>
#include <stdio.h>
#include "mex.h"
#include "matrix.h"

/* 1D Convolution operator */
void conv(double *x,
        int lx,
        double *h,
        int lh,
        double *y) {
    int i, j;
    double x0;
    for(i=lx;i<lx+lh-1;i++)
        x[i] = x[i-lx];
    for(i=0;i<lx;i++){
        x0 = 0;
        for(j=0;j<lh;j++){
            x0 = x0+x[i+j]*h[lh-1-j];
        }
        y[i] = x0;
    }
}

/* Main Function */
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])

{
    double *input, *h_row, *h_col, *output;
    double *row_in, *row_out, *col_in, *col_out;
    int Nx, Ny, Lh;
    int i, j, j0;
    
    if (nrhs!=3){
        mexErrMsgTxt("There must be 3 input parameters!");
        return;
    }
    input = mxGetPr(prhs[0]);
    h_row = mxGetPr(prhs[1]);
    h_col = mxGetPr(prhs[2]);
    Nx    = mxGetM(prhs[0]);
    Ny    = mxGetN(prhs[0]);
    Lh    = mxGetN(prhs[1]);

    plhs[0] = mxCreateDoubleMatrix(Nx,Ny,mxREAL);
    output  = mxGetPr(plhs[0]);

    for(i=0;i<Nx*Ny;i++)
        output[i] = input[i];
    if(Ny>1){
        row_in  = malloc((Ny+Lh-1)*sizeof(double));
        row_out = malloc(Ny*sizeof(double));
        for(i=0;i<Nx;i++){
            for(j=0;j<Ny;j++){
                row_in[j] = output[i+j*Nx];
            }
            conv(row_in,Ny,h_row,Lh,row_out);
            for(j=0;j<Ny;j++){
                output[i+j*Nx] = row_out[j];
            }
        }
        free(row_in);
        free(row_out);
    }
    if(Nx>1){
        col_in  = malloc((Nx+Lh-1)*sizeof(double));
        col_out = malloc(Nx*sizeof(double));
        for(j=0;j<Ny;j++){
            j0 = j*Nx;
            for(i=0;i<Nx;i++){
                col_in[i] = output[i+j0];
            }
            conv(col_in,Nx,h_col,Lh,col_out);
            for(i=0;i<Nx;i++){
                output[i+j0] = col_out[i];
            }
        }
        free(col_in);
        free(col_out);
    }
}