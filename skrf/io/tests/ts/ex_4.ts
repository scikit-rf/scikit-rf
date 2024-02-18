! 4-port S-parameter data
! Default impedance is overridden by [Reference]
! Data cannot be represented using 1.0 syntax
! Note that the [Reference] keyword arguments appear on a separate line
[Version] 2.0
# GHz S MA R 50
[Number of Ports] 4
[Reference]
50 75 0.01 0.01
[Number of Frequencies] 1 
[Network Data]
1   11 0 12 0 13 0 14 0
    21 0 22 0 23 0 24 0
    31 0 32 0 33 0 34 0
    41 0 42 0 43 0 44 0 