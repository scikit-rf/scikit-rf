## rectangular waveguide specific
#class DelayedTermination_TranslationMissalignment(ParametricStandard):
        #'''
        #A delayed rectangular waveguide termination with unknown flange
        #translation missalignment.

        #'''
        #def __init__(self, media,d,Gamma0,initial_offset= 1./10, **kwargs):
                #'''
                #takes:
                        #media: a Media type, with a RectangularWaveguide object
                                #for its tline property.
                        #d: distance to termination
                        #Gamma0: reflection coefficient off termination at termination
                        #initial_offset: initial offset guess, as a fraction of a,
                                #(the waveguide width dimension)
                        #**kwargs: passed to self.function
                #'''
                #wg = media.tline
                #kwargs.update({\
                        #'wg_I':wg,\
                        #'wg_II':wg,\
                        #'freq':media.frequency,\
                        #'d':d,\
                        #'Gamma0':Gamma0})

                #ParametricStandard.__init__(self, \
                        #function = rectangular_junction_centered,\
                        #parameters = {\
                                #'da':wg.a*initial_offset, \
                                #'db':wg.a*initial_offset},\
                        #**kwargs\
                        #)
#class DelayedShort_TranslationMissalignment(ParametricStandard):
        #'''
        #A delayed rectangular waveguide termination with unknown flange
        #translation missalignment.

        #'''
        #def __init__(self, media,d,initial_offset= 1./10, **kwargs):
                #'''
                #takes:
                        #media: a Media type, with a RectangularWaveguide object
                                #for its tline property.
                        #d: distance to termination
                        #Gamma0: reflection coefficient off termination at termination
                        #initial_offset: initial offset guess, as a fraction of a,
                                #(the waveguide width dimension)
                        #**kwargs: passed to self.function
                #'''
                #wg = media.tline
                #kwargs.update({\
                        #'wg_I':wg,\
                        #'wg_II':wg,\
                        #'freq':media.frequency,\
                        #'d':d,\
                        #'Gamma0':-1})

                #ParametricStandard.__init__(self, \
                        #function = rectangular_junction_centered,\
                        #parameters = {\
                                #'da':wg.a*initial_offset, \
                                #'db':wg.a*initial_offset},\
                        #**kwargs\
                        #)
#class Line_TranslationMissalignment(ParametricStandard):
        #'''
        #A  rectangular waveaguide matched line standard with unknown flange
        #translation missalignment.

        #'''
        #def __init__(self, media,d,initial_offset= 1./10, **kwargs):
                #'''
                #takes:
                        #media: a Media type, with a RectangularWaveguide object
                                #for its tline property.
                        #d: length of line [m]
                        #Gamma0: reflection coefficient off termination at termination
                        #initial_offset: initial offset guess, as a fraction of a,
                                #(the waveguide width dimension)
                        #**kwargs: passed to self.function

                #'''
                #wg = media.tline
                #kwargs.update({\
                        #'wg_I':wg,\
                        #'wg_II':wg,\
                        #'freq':media.frequency,\
                        #'d':d,\
                        #'Gamma0':0,\
                        #'nports':2,\
                        #})

                #ParametricStandard.__init__(self, \
                        #function = rectangular_junction_centered,\
                        #parameters = {\
                                #'da':wg.a*initial_offset, \
                                #'db':wg.a*initial_offset},\
                        #**kwargs\
                        #)
#class Line_UnknownLength_TranslationMissalignment(ParametricStandard):
        #'''
        #A  rectangular waveaguide matched line standard with unknown flange
        #translation missalignment and unknown length.

        #'''
        #def __init__(self, media,d,initial_offset= 1./10, **kwargs):
                #'''
                #takes:
                        #media: a Media type, with a RectangularWaveguide object
                                #for its tline property.
                        #d: guess for length of line [m]
                        #Gamma0: reflection coefficient off termination at termination
                        #initial_offset: initial offset guess, as a fraction of a,
                                #(the waveguide width dimension)
                        #**kwargs: passed to self.function

                #'''
                #wg = media.tline
                #kwargs.update({\
                        #'wg_I':wg,\
                        #'wg_II':wg,\
                        #'freq':media.frequency,\
                        #'Gamma0':0,\
                        #'nports':2,\
                        #})

                #ParametricStandard.__init__(self, \
                        #function = rectangular_junction_centered,\
                        #parameters = {\
                                #'da':wg.a*initial_offset, \
                                #'db':wg.a*initial_offset,
                                #'d':d
                                #},\
                        #**kwargs\
                        #)
#class Thru_TranslationMissalignment(ParametricStandard):
        #'''
        #A  rectangular waveaguide thru standard with unknown flange
        #translation missalignment.

        #'''
        #def __init__(self, media,initial_offset= 1./10, **kwargs):
                #'''
                #takes:
                        #media: a Media type, with a RectangularWaveguide object
                                #for its tline property.
                        #d: distance to termination
                        #Gamma0: reflection coefficient off termination at termination
                        #initial_offset: initial offset guess, as a fraction of a,
                                #(the waveguide width dimension)
                        #**kwargs: passed to self.function

                #'''
                #wg = media.tline
                #kwargs.update({\
                        #'wg_I':wg,\
                        #'wg_II':wg,\
                        #'freq':media.frequency,\
                        #'d':0,\
                        #'Gamma0':0,\
                        #'nports':2,\
                        #})

                #ParametricStandard.__init__(self, \
                        #function = rectangular_junction_centered,\
                        #parameters = {\
                                #'da':wg.a*initial_offset, \
                                #'db':wg.a*initial_offset},\
                        #**kwargs\
                        #)
#class Match_TranslationMissalignment(ParametricStandard):
        #'''
        #A match with unknown translation missalignment.
        #the initial guess for missalignment is [a/10,a/10], where a is the
        #waveguide width
        #'''
        #def __init__(self, media, initial_offset= 1./10 , **kwargs):
                #'''
                #takes:
                        #media: a Media type, with a RectangularWaveguide object
                                #for its tline property.
                        #initial_offset: initial offset guess, as a fraction of a,
                                #(the waveguide width dimension)
                        #**kwargs: passed to self.function
                #'''
                #wg = media.tline
                #kwargs.update({'wg':wg,'freq':media.frequency})

                #ParametricStandard.__init__(self, \
                        #function = translation_offset,\
                        #parameters = {'delta_a':wg.a*initial_offset, \
                                #'delta_b':wg.a*initial_offset},\
                        #**kwargs\
                        #)




#class DelayedTermination_UnknownLength_TranslationMissalignment(ParametricStandard):
        #'''
        #A known Delayed Termination with unknown translation missalignment.
        #the initial guess for missalignment defaults to [1/10,1/10]*a,
        #where a is the         waveguide width
        #'''
        #def __init__(self, media,d,Gamma0,initial_offset= 1./10, **kwargs):
                #'''
                #takes:
                        #media: a Media type, with a RectangularWaveguide object
                                #for its tline property.
                        #d: distance to termination
                        #Gamma0: reflection coefficient off termination at termination
                        #initial_offset: initial offset guess, as a fraction of a,
                                #(the waveguide width dimension)
                        #**kwargs: passed to self.function
                #'''
                #wg = media.tline
                #kwargs.update({\
                        #'wg_I':wg,\
                        #'wg_II':wg,\
                        #'freq':media.frequency,\
                        #'Gamma0':Gamma0})

                #ParametricStandard.__init__(self, \
                        #function = rectangular_junction_centered,\
                        #parameters = {'da':wg.a*initial_offset, \
                                                #'db':wg.a*initial_offset,\
                                                #'d':d},\
                        #**kwargs\
                        #)

#class RotatedWaveguide_UnknownLength(ParametricStandard):
        #'''
        #A rotated waveguide of unkown delay length.
        #'''
        #def __init__(self, media,d,Gamma0, **kwargs):
                #'''
                #takes:
                        #media: a Media type, with a RectangularWaveguide object
                                #for its tline property.
                        #d: distance to termination
                        #Gamma0: reflection coefficient off termination at termination
                        #initial_offset: initial offset guess, as a fraction of a,
                                #(the waveguide width dimension)
                        #**kwargs: passed to self.function
                #'''
                #wg_I = media.tline
                #wg_II = deepcopy(media.tline)
                #wg_II.a , wg_II.b = wg_I.b,wg_I.a

                #kwargs.update({\
                        #'wg_I':wg,\
                        #'wg_II':wg,\
                        #'freq':media.frequency,\
                        #'da':0,\
                        #'db':0,\
                        #'Gamma0':Gamma0})

                #ParametricStandard.__init__(self, \
                        #function = rectangular_junction_centered,\
                        #parameters = {'d':d},\
                        #**kwargs\
                        #)
