import skrf as rf

if True :
    def_fixture_s2p =   'Port1_path.s2p'                                         
    thru_s2p = 'thru.s2p'
    WG_MS = rf.Network(def_fixture_s2p, name = 'WG')['75-75.05GHz']
    thru = rf.Network(thru_s2p, name = 'WG')['75-75.05GHz']
    
    Ncascade = WG_MS ** thru

    print (Ncascade.nfmin_db, Ncascade.g_opt, Ncascade.rn)

    deembed = WG_MS.inv
    deembed ** Ncascade
    
    orig =  deembed ** Ncascade
    
    print (orig.nfmin_db, orig.g_opt, orig.rn)
    print (thru.nfmin_db, thru.g_opt, thru.rn)
    
    thru.nfmin_db
    thru.noise_freq.f
