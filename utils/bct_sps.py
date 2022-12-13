import numpy as np


def moment(x, fx, avg, mom=2):
    return np.sum(fx * (x - avg)**mom) / np.sum(fx)


class BCTDC():

    def __init__(self, japcIn):

        self.japc = japcIn
        self.validCycle = False
        self.chroma = 0.0

    def getValue(self):
        int_vec = self.japc.getParam("SPS.BCTDC.31832/Acquisition#totalIntensity")
        
        ts = self.japc.getParam('SPS.BCTDC.31832/Acquisition#samplingTime')
        
        self.time = np.linspace(0, (len(int_vec) - 1) * ts, len(int_vec))
        
        self.int_vec = int_vec
        if max(int_vec > 5):
            self.dpp_x, self.dpp_y = self.calculateMomSpread()
    
    
    def callBack(self, paramName, newValue):
        if max(newValue) > 5:
            self.validCycle = True
            int_vec = newValue
            
            ts = self.japc.getParam('SPS.BCTDC.31832/Acquisition#samplingTime')
            
            self.time = np.linspace(0, (len(int_vec) - 1) * ts, len(int_vec))
            
            self.int_vec = int_vec
            
            self.dpp_x, self.dpp_y = self.calculateMomSpread()
            
            
    def calculateMomSpread(self):
        # Get tune from LSA
        # calculate tune range
        # maybe do a fit
        # save pickles

        tune_lsa = self.japc.getParam('rmi://virtual_sps/SPSBEAM/QH')
        beam_out = self.japc.getParam('SX.BEAM-OUT-CTML/ControlValue#controlValue') - 5
        beam_in = self.japc.getParam('SIX.MC-CTML/ControlValue#controlValue')
        flat_top = self.japc.getParam('SX.S-FTOP-CTML/ControlValue#controlValue')

        ft_time = [flat_top, beam_out]
        ft_index = [list(tune_lsa[0]).index(ft_time[0]), list(tune_lsa[0]).index(ft_time[1])]
        
        delta_tune = tune_lsa[1][ft_index[1]] - tune_lsa[1][ft_index[0]]
        
        self.tune_time = tune_lsa[0][ft_index[0]:ft_index[1]] - beam_in
        self.tune = tune_lsa[1][ft_index[0]:ft_index[1]]

        self.time_int = np.linspace(flat_top - beam_in, beam_out - beam_in, 200)
        
        self.bct_int = np.interp(self.time_int, self.time, self.int_vec)
        
        inv_bct = self.bct_int[0] - self.bct_int
        didt = np.gradient(inv_bct, 5e-3)
        
        
        if not self.chroma == 0:

            self.tune_int = np.interp(self.time_int, self.tune_time, self.tune)    
            
            dtune = np.linspace(-1 * delta_tune / 2, delta_tune / 2, len(self.time_int), endpoint=True)
            dpp = dtune / (self.chroma * 26.66) * 1e3
                          
            self.pdf_dpp = didt / np.gradient((self.tune_int), self.time_int)
#            self.pdf_dpp = self.pdf_dpp / max(self.pdf_dpp)
            
            self.dpp_mean = np.average(dpp, weights=didt)
            print(self.dpp_mean)
            self.dpp_rms = moment(dpp, didt, self.dpp_mean) 
            
            
            return dpp, didt
        else:
            return [], []
            
