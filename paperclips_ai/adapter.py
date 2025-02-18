"""
An adapter which handles retrieving data from the game and taking actions within it.
"""

import logging

from selenium import webdriver
from selenium.webdriver.common.by import By


class WebElement:
    """A clickable web element."""
    def __init__(self, driver, name):
        """Initializes a clickable web element."""
        self.name = name
        self.driver = driver
        self.elem = driver.find_element(By.ID, name)

    def click(self, n: int = 1):
        """Clicks the element n times.
        
        Args:
            n: The number of times to click the element.
        """
        for _ in range(n):
            try:
                # this is faster than the normal click method
                self.driver.execute_script("arguments[0].click();", self.elem)
            except Exception as e:      # pylint: disable=W0718
                print("Click script failed to execute:", e)

    @property
    def text(self) -> str:
        """The text of the element."""
        return self.elem.text

    @property
    def value(self) -> float:
        """The value of the element."""
        text = self.elem.text
        if text:
            return float(text.replace(",", ""))
        return float(0)

class EmptyWebElement(WebElement):
    """A class representing a placeholder web element that serves no real purpose."""
    def __init__(self, name): # pylint: disable=W0231
        self.name = name

    def click(self, n: int = 1):
        pass

    @property
    def text(self) -> str:
        return ""

    @property
    def value(self) -> float:
        return 0


class WebAdapter:
    """An adapter for interacting with the game."""
    def __init__(self, webpage):
        """Initializes the adapter.

        Args:
            webpage: The url of the game.
        """
        self.logger = logging.getLogger('paperclips_ai.WebAdapter')

        # requires firefox and driver to be installed
        self.driver = webdriver.Firefox()

        # open the website
        self.driver.get(webpage)
        self.logger.info("Connection initialized")

        self.clips_el = self.get_elem_by_id('clips')

    def get_elem_by_id(self, name):
        """Get a clickable element by its id.
        
        Args:
            name: The id of the element.
        """
        return WebElement(self.driver, name)
