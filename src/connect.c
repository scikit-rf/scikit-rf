

#include <string.h>
#include <stdlib.h>
#include <complex.h>

void initconnect()
{    } 

/*
 ==============================
 Array flattening convention: 
 If A has shape=(x,y,z)
 A[i][j][k] = A[i*y*z+j*z+k]
 ============================== 
 */

void innerconnect_s(double *freq, int nFreq, complex *A, int nA, int k, int l, complex *D, int nD){
    /*
     freq - 1D array of freqeuncy values
     nFreq - length(freq)
     A - nFreq x nA x nA complex-valued array of composite network
     nA - number of ports for composite network A
     k - port index to connect on original network A (starts at 0)
     nB - number of ports for network B
     l - port index to connect on original network B (starts at 0)
     D - nFreq x nD x nD complex-valued array (to be filled with resulting connected network)
     */
    
    //C = connected matrix (temporary)
    complex *C = malloc(nFreq*nA*nA*sizeof(complex));
    
    //populate connected matrix
    int q, r, s;
    for (s = 0; s < nFreq; s++){ //for each freq point
        for(q = 0; q < nA; q++){ //for each row of A
            for(r = 0; r < nA; r++){ //for each column of A
                //write connected element into C
                C[s*nA*nA+q*nA+r] = A[s*nA*nA+q*nA+r] + ( A[s*nA*nA+k*nA+r]*A[s*nA*nA+q*nA+l]*(1-A[s*nA*nA+l*nA+k]) + A[s*nA*nA+l*nA+r]*A[s*nA*nA+q*nA+k]*(1-A[s*nA*nA+k*nA+l]) + A[s*nA*nA+k*nA+r]*A[s*nA*nA+l*nA+l]*A[s*nA*nA+q*nA+k] + A[s*nA*nA+l*nA+r]*A[s*nA*nA+k*nA+k]*A[s*nA*nA+q*nA+l])/( (1-A[s*nA*nA+k*nA+l])*(1-A[s*nA*nA+l*nA+k]) - A[s*nA*nA+k*nA+k]*A[s*nA*nA+l*nA+l] );
            }
        }
    }
        
    //create result matrix -- remove rows and cols 'k' and 'l'
    int curRow, curCol; //"pointer" to next S-param to write
    for(s = 0; s < nFreq; s++){ //for each freq point
        curRow = 0; curCol = 0; //init pointer to S11
        for(q = 0; q < nA; q++){ // for each row
            if(q != k && q != l){ // if not on row 'k' or 'l'
                curCol = 0; //move column pointer back to port 1
                for(r = 0; r < nA; r++){ // for each column
                    if(r != k && r != l){ // if not on column 'k' or 'l'
                        //copy S-param from connected matrix into result matrix and increment column pointer
                        D[s*nD*nD+curRow*nD+curCol++] = C[s*nA*nA+q*nA+r];                        
                    }
                }
                curRow++; //hit end of column, move to next row
            }      
        }        
    }
    
    free(C); return;
}


void connect_s(double *freq, int nFreq, complex *A, int nA, int k, complex *B, int nB, int l, complex *D, int nD){ 
    /*
     freq - 1D array of freqeuncy values
     nFreq - length(freq)
     A - nFreq x nA x nA complex-valued array of network A's s-parameters
     nA - number of ports for network A
     k - port index to connect on A (starts at 0)
     B - nFreq x nB x nB complex-valued array of network B's s-parameters
     nB - number of ports for network B
     l - port index to connect on B (starts at 0)
     D - nFreq x nD x nD complex-valued array (to be filled with resulting connected network)
     nD - number of ports for resulting network (should be nA+nB-2)
     */
    
    //C = zeros(nFreq, nC, nC), for building composite matrix
    int nC = nA + nB;
    complex *C = calloc(nC*nC*nFreq, sizeof(complex)); 
    
    //build composite matrix
    int q,r;
    for(q = 0; q < nFreq; q++){ //for each freq point
        //copy A into upper left of C, row by row
        for(r = 0; r < nA; r++){
            //C[freq][r][1:nA] = A[freq][r][1:nA]
            memcpy(&C[q*nC*nC+r*nC], &A[q*nA*nA+r*nA], nA*sizeof(complex));
        }
        //copy B into lower right of C, row by row
        for(r = nA; r < nC; r++){
            //C[freq][r][nA+1:nA+1+nB] = B[freq][r][1:nB]
            memcpy(&C[q*nC*nC+r*nC+nA], &B[q*nB*nB+(r-nA)*nB], nB*sizeof(complex));
        }        
    }
    
    innerconnect_s(freq, nFreq, C, nC, k, l+nA, D, nD);
    free(C); return;
}
