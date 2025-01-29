import logging
import os

from selenium import webdriver
from selenium.webdriver.common.by import By


class WebElement:
    def __init__(self, driver, name):
        self.name = name
        self.driver = driver
        self.elem = driver.find_element(By.ID, name)

    def click(self, n=1):
        for _ in range(n):
            try:
                self.driver.execute_script("arguments[0].click();", self.elem)
            except Exception as e:
                print("exception:", e)

    @property
    def text(self):
        return self.elem.text

    @property
    def value(self):
        text = self.elem.text
        if text:
            return float(text.replace(",", ""))
        return 0

class WebAdapter:
    def __init__(self, webpage):
        # path to web driver executable
        self.logger = logging.getLogger('paperclips_ai.WebAdapter')
        package_dir = os.path.dirname(os.path.abspath(__file__))
        driver_path = os.path.join(package_dir, 'resources/geckodriver.exe')

        self.driver = webdriver.Firefox()

        # open the website
        self.driver.get(webpage)
        self.logger.info("Connection initialized")

        self.clips_el = self.get_elem_by_id('clips')

    def get_elem_by_id(self, name):
        return WebElement(self.driver, name)

    def get_returns(self):
        return float(self.clips_el.value)