
'''
provides the mobius class.
'''

from numpy import array, reshape, shape
class mobius(object):
    def __init__(self, h):
        h = array(h)

        if len(shape(h)) == 1:
            h=reshape(h, (2,2))

        self.a = h[0,0]
        self.b = h[0,1]
        self.c = h[1,0]
        self.d = h[1,1]

        self.h = h

    def transform(self,w):
        return (self.a*w +self.b)/(self.c*w+self.d)
    def itransform(self,z):
        return (self.d*z-self.b)/(-self.c*z+self.a)




def mobiusTransformation(m, a):
    '''
    returns the unique maping function between m and a planes which are
    related through the mobius transform.

    takes:
            m: list containing the triplet of points in m plane m0,m1,m2
            a: list containing the triplet of points in a plane a0,a1,a2

    returns:
            a (m) : function of variable in m plane, which returns a value
                    in the a-plane
    '''
    m0,m1,m2 = m
    a0,a1,a2 = a
    return lambda m: (a0*a1*m*m0 + a0*a1*m1*m2 + a0*a2*m*m2 + a0*a2*m0*m1 +\
     a1*a2*m*m1 + a1*a2*m0*m2 - a0*a1*m*m1 - a0*a1*m0*m2 - a0*a2*m*m0 -\
     a0*a2*m1*m2 - a1*a2*m*m2 - a1*a2*m0*m1)/(a0*m*m2 + a0*m0*m1 + a1*m*m0\
      + a1*m1*m2 + a2*m*m1 + a2*m0*m2 - a0*m*m1 - a0*m0*m2 - a1*m*m2 - \
      a1*m0*m1 - a2*m*m0 - a2*m1*m2)
