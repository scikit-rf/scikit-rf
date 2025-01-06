! 4-port S-parameter data
! Default impedance is overridden by the [Reference] keyword arguments
! Note that [Reference] arguments are split across two lines
! Data cannot be represented using 1.0 syntax
[Version] 2.0
# GHz S MA R 50
[Number of Ports] 4
[Number of Frequencies] 2
[Reference] 50 75
0.01 0.01
[Matrix Format] Lower
[Network Data]
5.00000 0.60 161.24 !row 1
 0.40 -42.20 0.60 161.20 !row 2
 0.42 -66.58 0.53 -79.34 0.60 161.24 !row 3
 0.53 -79.34 0.42 -66.58 0.40 -42.20 0.60 161.24 !row 4 
6.00000 0.60 161.24 !row 1
 0.40 -42.20 0.60 161.20 !row 2
 0.42 -66.58 0.53 -79.34 0.60 161.24 !row 3
 0.53 -79.34 0.42 -66.58 0.40 -42.20 0.60 161.24 !row 4 