import random
import logging
import numpy as np
import pandas as pd
import os


class ManaMarketDataLoader:
    """
    A class to handle market data loading and processing from MANA
    """

    def __init__(self,  start_date, end_date, stock_list, start_date_pct_range=0.0, end_date_pct_range=1.0,  dir_path='mana_data', load_data=True):
        """
        Initialize the ManaMarketDataLoader
        @param start_date: Start date of the data
        @param end_date: End date of the data
        @param stock_list: List of stock symbols
        @param start_date_pct_range: Percentage range of the data to load from the start of the data
        @param end_date_pct_range: Percentage range of the data to load from the end of the data
        @param dir_path: Path to the directory containing the data
        @param load_data: Whether to load the data
        """
        self.dir_path = dir_path
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist")
        

        if start_date is None and end_date is None:
            calculate_date_range()
        else:
            self.start_date = start_date
            self.end_date = end_date

        if load_data:
            self.load_bars_from_mana()




        def calculate_date_range(self):
            """
            Calculate the date range of the data
            """
            self.first_date = self.bars_df['date'].min()
            self.last_date = self.bars_df['date'].max()
            self.start_date = self.first_date + (self.last_date - self.first_date) * self.start_date_pct_range
            self.end_date = self.last_date - (self.last_date - self.first_date) * self.end_date_pct_range


        def load_bars_from_mana(self):
            """
            Load the bars from the mana data 
            """
            