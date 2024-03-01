from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import timeit
import time
import numpy as np

class ActionInterface:
    def __init__(self, driver):
        self.driver = driver
        self.make_clip_bttn = self.driver.find_element(By.ID, "btnMakePaperclip")
        self.buy_wire_bttn = self.driver.find_element(By.ID, "btnBuyWire")
    
    def makePaperClip(self):
        self.driver.execute_script("arguments[0].click();", self.make_clip_bttn)
    
    def buyWire(self):
        self.driver.execute_script("arguments[0].click()", self.buy_wire_bttn)