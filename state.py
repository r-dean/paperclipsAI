from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import timeit
import time
import numpy as np

class State:
    def __init__(self, driver):
        features = 10
        self.state = np.zeros(features)

        # intialize features
        self.clips = 0
        self.funds = 0
        self.unsold = 0
        self.wire = 0
        self.price = 0
        self.demand = 0
        self.wire_cost = 0
        self.autoclip_available = 0
        self.autoclip = 0
        self.autoclip_cost =0

        # find elements
        self.driver = driver
        self.clips_el = driver.find_element(By.ID, "clips")
        self.funds_el = driver.find_element(By.ID, "funds")
        self.unsold_el = driver.find_element(By.ID, "unsoldClips")
        self.wire_el = driver.find_element(By.ID, "wire")
        self.price_el = driver.find_elements(By.ID, "margin")
        self.demand_el = driver.find_elements(By.ID, "demand")
        self.wirecost_el = driver.find_elements(By.ID, "wireCost")
        self.autoclip_el = driver.find_elements(By.ID, "clipmakerLevel2")
        self.autoclip_price_el = driver.find_elements(By.ID, "clipperCost")

        self.updateState()

    '''Read data from webpage to determine current state'''
    def updateState(self):
        # get values
        self.clips = float(self.clips_el.text.replace(",", ""))
        self.funds = float(self.funds_el.text.replace(",", ""))
        self.unsold = float(self.unsold_el.text.replace(",", ""))
        self.wire = float(self.wire_el.text.replace(",",""))
        self.price = float(self.price_el[0].text.replace(",",""))
        self.demand = float(self.demand_el[0].text.replace(",",""))
        self.wire_cost = float(self.wirecost_el[0].text.replace(",",""))
        autoclip_text = self.autoclip_el[0].text
        if autoclip_text:
            self.autoclip_available = 1
            self.autoclip = float(autoclip_text.replace(",",""))
            self.autoclip_cost = float(self.autoclip_price_el[0].text.replace(",",""))

        # update state vector
        self.state[0] = self.clips
        self.state[1] = self.funds
        self.state[2] = self.unsold
        self.state[3] = self.demand
        self.state[4] = self.price
        self.state[5] = self.wire
        self.state[6] = self.wire_cost
        if autoclip_text:
            self.state[7] = self.autoclip_available
            self.state[8] = self.autoclip
            self.state[9] = self.autoclip_cost
        

    '''Get a stacked state-action vector for determining whether to wait, make clips, or buy wire'''
    def getThreeActionVect(self):
        
        features = 8
        state_vec = np.zeros(features)
        
        state_vec[0] = 0        # bias
        state_vec[1] = self.clips
        state_vec[2] = self.funds
        state_vec[3] = self.unsold
        state_vec[4] = self.price
        state_vec[5] = self.demand
        state_vec[6] = self.wire 
        state_vec[7] = self.wire_cost

        state_vec = (state_vec - np.mean(state_vec)) / np.std(state_vec)
        state_vec[0] = 1

        state_vec = np.vstack([state_vec] * 3)
        return state_vec 

    def getAccumulatedReturn(self):
        return self.clips

    def updateAndCalculateReward(self):
        old_return = self.clips
        self.updateState()
        return self.clips - old_return

    def __repr__(self):
        str1 = f"Clips: {self.clips}\tFunds: {self.funds}\tUnsold: {self.unsold}\nPrice: {self.price}\tDemand: {self.demand}\nWire: {self.wire}\tWire Cost: {self.wire_cost}\n"
        if self.autoclip_available:
            str2 = f"Auto Clippers: {self.autoclip}\tAuto Clipper Cost: {self.autoclip_cost}"
            str1 = str1 + str2
        return str1
