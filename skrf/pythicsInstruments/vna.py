#import skrf as rf
from skrf.vi import vna as mv_vna
from skrf import Network
import pdb
import multiprocessing

class Private():
    def __init__(self):
        self.network = None
        self.vna = None
        self.ax = None
        self.logger = multiprocessing.get_logger()
        self.plot_format = 'Magnitude[dB]'
private = Private()


def connect_to_vna(text_gpib_address,text_gpib_timeout,choice_vna_model,\
        button_connect_to_vna,**kwargs):

    vna_class_dict = {\
            'HP8510C':mv_vna.HP8510C,\
            'HP8720':mv_vna.HP8720,\
            }
    if private.vna is not None:
        private.vna.close()
        private.vna = None
        button_connect_to_vna.label='Disconnected.'
        private.logger.error('VNA Disconnected ')
    else:
        try:
            private.vna = vna_class_dict[choice_vna_model.value]\
                    (int(text_gpib_address.value), \
                    timeout=int(text_gpib_timeout.value))
            button_connect_to_vna.label='Connected.'
            private.logger.info('VNA connected.')
        except:
            private.logger.error('VNA failed to load. ')
            button_connect_to_vna.label='Failed.'


def open_file(file_dialog, file_dialog_result,mpl_plot, **kwargs):
    file_dialog_result.value = file_dialog.open()
    private.network = Network(file_dialog_result.value)
    update_plot(mpl_plot)

def clear_file_dialog( file_dialog_result,**kwargs):
    file_dialog_result.value = ''

def save_file(file_dialog, file_dialog_result,**kwargs):
    if private.network is not None:
        if file_dialog_result.value is '':
            file_dialog_result.value = file_dialog.save()
        private.network.write_touchstone(file_dialog_result.value)
        private.logger.info('file: %s written.'%file_dialog_result.value)
    else:
        private.logger.error('No Network in memory')



def get_one_port(m,n, mpl_plot, *args, **kwargs):
    private.logger.info('Getting S%i%i'%(m,n))
    if private.vna is None:
        private.logger.error('VNA not connected')
    else:
        private.network = private.vna.__getattribute__('s%i%i'%(m,n))
        update_plot(mpl_plot    )




def get_two_port(mpl_plot, **kwargs):
    if private.vna is None:
        private.logger.error('VNA not connected')
    else:
        private.network = private.vna.two_port
        update_plot(mpl_plot)

def get_switch_terms(file_dialog_forward_switch_terms,\
        file_dialog_reverse_switch_terms, **kwargs):
    private.switch_terms = private.vna.switch_terms
    file_dialog_forward_switch_terms.title = 'Save Forward Switch Term'
    file_dialog_forward_switch_terms.filename = 'forward_switch_term.s1p'
    private.switch_terms[0].write_touchstone(\
            file_dialog_forward_switch_terms.save())
    file_dialog_reverse_switch_terms.title = 'Save Reverse Switch Term'
    file_dialog_reverse_switch_terms.filename = 'reverse_switch_term.s1p'
    private.switch_terms[1].write_touchstone(\
            file_dialog_reverse_switch_terms.save())


def get_s11(mpl_plot,  **kwargs):
    return get_one_port(1,1,mpl_plot)
def get_s12(mpl_plot, **kwargs):
    return get_one_port(1,2,mpl_plot)
def get_s21(mpl_plot, **kwargs):
    return get_one_port(2,1,mpl_plot)
def get_s22(mpl_plot, **kwargs):
    return get_one_port(2,2,mpl_plot)

def clear_plot(mpl_plot, **kwargs):
    mpl_plot.clear()
    mpl_plot.show()


def change_plot_format(radio_plot_format,mpl_plot, **kwargs):
    private.plot_format = radio_plot_format.value
    clear_plot(mpl_plot)
    update_plot(mpl_plot)

def update_plot(mpl_plot, **kwargs):
    plot_format = private.plot_format
    network = private.network
    type_dict = {\
            'Magnitude[dB]':'s_db',\
            'Phase[deg]':'s_deg',\
            'Smith Chart':'',\
            }
    #FUTURE FIX: this will work once bug is fixed in pythics
    #private.ax = mpl_plot.get_axes()
    #private.network.plot_s_db(m-1,n-1, ax = private.ax)
    for m in range (network.number_of_ports):
        for n in range(network.number_of_ports):
            mpl_plot.plot(network.frequency.f_scaled, \
                    network.__getattribute__(type_dict[plot_format])[:,m,n]\
                    , label= network.name )
            mpl_plot.set_xlabel('Frequency[%s]'%(network.frequency.unit))
            mpl_plot.set_ylabel(plot_format)
    mpl_plot.set_title(plot_format)
    mpl_plot.legend()
    mpl_plot.axis('tight')
    mpl_plot.show()
