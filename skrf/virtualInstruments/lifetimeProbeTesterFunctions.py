import skrf as rf
from skrf.virtualInstruments import lifetimeProbeTester as lpt
import os

def setup():
    l = lpt.LifeTimeProbeTester()
    l.uncontact_gap = .03
    l.stage.delay = .4
    l.stage.velocity = 1
    l.stage.acceleration = 1
    l.stage2.velocity = 1
    l.stage2.acceleration = 1
    l.step_increment =.001
    l.raiseup_overshoot=.3
    return l

def run_cal(l, contact_force,  dir='',one_on_top = True):
    try:
        os.mkdir(dir)
    except(OSError):
        print 'directory exists, overwriting files'
    os.chdir(dir)

    l.contact_force = contact_force

    for line in range(5):
        l.contact_force = contact_force - 5
        l.contact_sloppy()
        l.contact_force = contact_force
        l.contact()
        l.read_network()
        l.read_network(name='ds%i'%(line+1), write=True)
        l.uncontact_sloppy()
        #l.uncontact()
        if one_on_top:
            l.move_left()
        else:
            l.move_right()
    if one_on_top:
        l.move_right(5)
    else:
        l.move_left(5)

    l.ntwk_history = []
    l.save_history()
    l.clear_history()
    os.chdir('..')


def make_cal(dir):
    ideal_dir = '../cpw_ideals/convertedByAlex/'
    prefix=''
    raw = rf.load_all_touchstones(dir, f_unit = 'ghz')
    all_ideals = rf.load_all_touchstones(ideal_dir)
    freq = raw[raw.keys()[0]].frequency
    cal = rf.Calibration( \
            type = 'one port',\
            is_reciprocal = True,\
            frequency = freq, \
            measured = [ raw[prefix +'ds'+str(k)] for k in range(1,6)],\
            ideals =  [ all_ideals['ds'+str(k)] for k in range(1,6)]
            )
    return cal

def cycle(l, slide_dist = 30, total_num_beatings = 2, beatings_per_spot=1):
    l.contact_sloppy()
    l.uncontact_sloppy()
    l.raiseup()
    l.move_right(slide_dist)
    l.beat_probe_up(total_num_beatings,beatings_per_spot,False)
    l.move_left(slide_dist)
    l.contact_sloppy()
    l.uncontact_sloppy()




'''
def life_time_test(l,landing_per_cal=1000, beatings_per_spot=100,\
        num_landings=10,contact_force=20):
                for landing_num in range(0,int(landings_per_cal* num_landings),landing_per_cal):
                        run_cal(l, contact_force, landing_num)

                        l.raiseup()
                        l.stage2.relative_po
                        l.set_position_upper_limit()

                        for spot_num in int(landing_per_cal/landings_per_spot):
                                for landing_num in range(landings_per_spot):
                                        l.contact_sloppy()
                                        l.uncontact_sloppy()


                        move_back_from_gold()
                        l.set_position_upper_limit()

def move_to_gold_and_back(l):
        cpw_spacing = .145
        l.raiseup_overshoot = .3
        for contact_num  in range(4):
                run_cal(l, 20, '%i'%(100*contact_num))
                l.raiseup()
                horizontal_distance = -(50 +contact_num)* cpw_spacing
                l.stage2.position_relative = -horizontal_distance
                for land_n in range(2):
                        l.
'''


vna = rf.Network('../chen/02-01-2011/0831_VNACal/vna.s2p')
