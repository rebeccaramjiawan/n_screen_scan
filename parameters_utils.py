#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 14:23:06 2018

@author: fvelotti
"""
import numpy as np
import sys
import logging
import pickle


class Parameter_utils():
    """
    High level API for handling set, get and subscribe of pyJAPC.
    The tricky part is that we use a give signal for subscription and
    the rest is handled with simple get and set.
    Inside the callBack function passed for the subscription, the getParam
    is done and value set as class variable __observed_data__.

    Parameters
    ----------
    japcIn : pyJapc
        pyJapc instance as used in the main class.
    """

    def __init__(self, japcIn, observer='', actor=''):

        self.japc = japcIn
        self.userName = None
        self.observer = observer
        self.actor = actor
        self.validCycle = False
        self.allOK = True
        self.is_calib = False

    def checkGet(getParam):

        def getParamWrapper(self, *args, **kwrds):
            try:
                data, info = getParam(self, *args, **kwrds)
            except:
                print("Error Get: ", sys.exc_info()[0])
                self.allOK = False
                data, info = None, {'selector': '', 'acqStamp': ''}
            return data, info

        return getParamWrapper

    def checkSet(setParam):

        def setParamWrapper(self, *args, **kwrds):
            try:
                setParam(self, *args, **kwrds)
                logging.debug('Set sent')
            except:
                print("Error Set: ", sys.exc_info()[0])
                self.allOK = False

        return setParamWrapper

    def save_initial_data(self):
        data_actor, info = self.actor.getParameter()
        self.file_name_ini = f'initial_{self.actor.simp_name}_settings_{str(info["acqStamp"]).replace(" ", "_")}.p'
        pickle.dump(data_actor, open(self.file_name_ini, 'wb'))

    def load_initial_data(self):
        data_to_set = pickle.load(open(self.file_name_ini, 'rb'))
        self.setParameter(self.actor, data_to_set)

    def callBack(self, paramName, newValue):
        """
        CallBack function for subscription. Subscribes to the passed
        paramName and sets varius class variables. Makes a cycle to be
        validated as valid, gets a time stamp. It also assigns to
        __observed_data__ the obtained parameters from the get method.


        Parameters
        ----------
        paramName : str
            Parameter name of the field to subscribe to
        newValue : obj
            Variable containing the field of property return of the
            subscription
        """
        observed_data, info = self.getParameter(self.observer)
        selector = info['selector']
        # Give valid cycle info for one get only
        self.validCycle = self.checkValidCycle() and self.checkSetValue() and selector == self.userName

        if self.validCycle:
            self.observed_data = observed_data
            self.timeStamp = info['acqStamp']
            self.actor_data, _ = self.getParameter(self.actor)

    def setScanLimits(self, dictScan):
        """Set limits of the of the variable to scan passed via the
        dictScan.

        Parameters
        ----------
        dictScan : dict
            Dictionary of the limits and info. It has to contain the keys:
            stats, start, end and steps.
        """
        self.scanSettings = dictScan
        self.stats = int(self.scanSettings['stats'])
        self.scanArray = np.linspace(self.scanSettings['start'],
                                     self.scanSettings['end'],
                                     int(self.scanSettings['steps']),
                                     endpoint=True)

    @checkSet
    def setParameter(self, actor, value):
        """Set parameters using the classic japc setParam method

        Parameters
        ----------
        value : object depending on the device (list, float, int...)
            Value that will be send to the HW via the set command
        """
        # This is a simple set - if need to set for virtual parameters (knobs) this is a bit more complex
        logging.debug('setParameter method called')
        if isinstance(actor, str):
            logging.debug('String passed to set method')
            self.japc.setParam(actor, value)
        else:
            logging.debug('Object passed to set method')
            actor.setParameter(value)

    @checkGet
    def getParameter(self, observer, getHeader=True, **kwrds):
        """Method to get parameter __observer__. It also handles errors as
        thrown by the getParam of pyJapc.

        Parameters
        ----------
        observer : str
            [server]device/property#field for needed device.
        """
        if isinstance(observer, str):
            data, info = self.japc.getParam(observer, getHeader=getHeader, **kwrds)
        else:
            data, info = observer.getParameter(getHeader=getHeader, **kwrds)
        return data, info

    def checkValidCycle(self):
        # TODO: need to implement logic to check if cycle is valid (BCT...)
        return True

    def checkSetValue(self):
        """__Not yet implemented__: this should check that the set value
        actually corresponds to what is in the HW.

        Parameters
        ----------
        demanded :
            demanded
        """
        # TODO: need to add test to check if set is there with 1% tolerance
        return True

