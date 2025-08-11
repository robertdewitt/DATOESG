import os
import datetime as dt
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union
from scipy import stats

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
import logging

logger = logging.getLogger(__name__)


class ManaMarketDataLoader:
    """
    Load and stream **MANA** minute‑bar parquet data.

    @param start_date: *Optional.* Lower date bound (``YYYY‑MM‑DD`` or
        :class:`date`). ``None`` ⇒ dataset minimum.
    @param end_date:   *Optional.* Upper date bound. ``None`` ⇒ dataset maximum.
    @param stock_list: *Optional.* Iterable of tickers; ``None`` or empty ⇒ all.
    @param dir_path:   Root folder that contains the parquet lake.
    @param data_set:   Sub‑folder under *dir_path* holding the files.
    @param load_data:  If *True*, immediately opens the dataset and prepares
        filters.  Set *False* if you want to postpone IO.
    """
    def __init__(
        self,
        start_date: Optional[Union[str, dt.date]] = None,
        end_date:   Optional[Union[str, dt.date]] = None,
        stock_list: Optional[Sequence[str]] = None,
        *,
        dir_path: str = "mana_data",
        data_set: str = "bar_data_parquet",
        load_data: bool = True,
    ) -> None:
        
        self.dir_path = dir_path
        self.data_set = data_set
        self.data_set_path = os.path.join(self.dir_path, self.data_set)
        if not os.path.exists(self.data_set_path):
            raise FileNotFoundError(f"Data‑set path '{self.data_set_path}' does not exist.")

        self.stock_list: List[str] = list(stock_list) if stock_list else []
        self._user_start = self._coerce_date(start_date)
        self._user_end   = self._coerce_date(end_date)
        
        # will be filled later
        self.ds_parquet: Optional[ds.Dataset] = None
        self._data_first: Optional[dt.date] = None
        self._data_last: Optional[dt.date] = None
        self.start_date: Optional[dt.date] = None
        self.end_date: Optional[dt.date] = None
        self.num_dates: Optional[int] = None
        self._date_filter: Optional[ds.Expression] = None
        self._sym_filter: Optional[ds.Expression] = None
        self._filter: Optional[ds.Expression] = None
        self.num_minutes: Optional[int] = None
        self._all_stock_list = None

        if load_data:
            self.load_bars_from_mana()

    @staticmethod
    def _coerce_date(val: Optional[Union[str, dt.date]]) -> Optional[dt.date]:
        """
        Return *val* as a :class:`date` or *None* unchanged.
        @param val: *Optional.* Date string or :class:`date` object.
        @return: *Optional.* :class:`date` or *None*.
        """
        if val is None:
            return None
        if isinstance(val, dt.date):
            return val
        return dt.datetime.strptime(val, "%Y-%m-%d").date()


    def load_bars_from_mana(self) -> None:
        """
        Open dataset and discover raw min / max trading days (strings ``YYYYMMDD``).
        @return: *None*.
        """
        # load parquet dataset
        self.ds_parquet = ds.dataset(
            self.data_set_path,
            format="parquet"
            )

        # Get unique dates from the partition structure
        date_strings: set[str] = set()
        for rel in self.ds_parquet.files:
            for part in Path(rel).parts:
                if part.startswith("date="):
                    date_strings.add(part.split("=", 1)[1])
                    break
        
        if len(date_strings) > 0:
            date_strings = {dt.datetime.strptime(s, "%Y%m%d").date() for s in date_strings} 
            self._data_first = min(date_strings)
            self._data_last  = max(date_strings)
            self.start_date = self._data_first
            self.end_date = self._data_last
            self.num_dates = len(date_strings)
        else:
            raise ValueError("No date information found in dataset")

        # get unique symbols
        for rel in self.ds_parquet.files:
            for part in Path(rel).parts:
                if part.startswith("symbol="):
                    self.all_stock_list.append(part.split("=", 1)[1])
                    break
        

    def _compute_effective_bounds(self, start_pct: float = 0.0, end_pct: float = 1.0) -> None:
        """
        Calculate *start_date*, *end_date* and build Arrow filter.
        @param start_pct: *Optional.* Fraction *f* (0 ≤ f ≤ 1) shifting the start, default=0.0
        @param end_pct: *Optional.* Fraction *g* (0 ≤ g ≤ 1) limiting the span, default=1.0
        @return: *None*.
        """
        if self.ds_parquet is None:
            raise RuntimeError("mkt_data_mana:_compute_effective_bounds: Dataset not loaded.")

        # 1) honour user explicit bounds
        first = self._data_first if self._user_start is None else max(self._data_first, self._user_start)
        last  = self._data_last  if self._user_end   is None else min(self._data_last,  self._user_end)
        if first > last:
            raise ValueError("mkt_data_mana:_comput_effective_bounds: start_date exceeds end_date after applying bounds.")

        # 2) Calculate total available days
        total_days = (last - first).days
        
        # 3) Calculate start and end dates based on percentages of total range
        start_offset = int(round(total_days * start_pct))
        end_offset = int(round(total_days * end_pct))
        
        self.start_date = first + dt.timedelta(days=start_offset)
        self.end_date = first + dt.timedelta(days=end_offset)
        
        # Ensure we don't exceed the last available date
        if self.end_date > last:
            self.end_date = last
            
        # Debug output for percentage calculations
        actual_days = (self.end_date - self.start_date).days
 
        # 4) Build Arrow predicate based on partitioning
        # Find field indices for date and symbol
        schema = self.ds_parquet.schema
        date_field_idx = None
        symbol_field_idx = None
        
        for i, field in enumerate(schema):
            if field.name == "date" and date_field_idx is None:
                date_field_idx = i
            elif field.name == "symbol" and symbol_field_idx is None:
                symbol_field_idx = i
        
        # Date is in the data - check its type
        if date_field_idx is not None:
            try:
                lower = pa.scalar(self.start_date, type=pa.date32())
                upper = pa.scalar(self.end_date, type=pa.date32())         
                self._date_filter = (ds.field(date_field_idx) >= lower) & (ds.field(date_field_idx) <= upper)
            except Exception as e:
                print(f"Warning: Could not create date filter: {e}")
                self._date_filter = None
        else:
            print("Warning: 'date' field not found in schema, skipping date filter")
            self._date_filter = None

        if self._date_filter is not None:
            if self.stock_list and symbol_field_idx is not None:
                self._sym_filter = ds.field(symbol_field_idx).isin(self.stock_list)
                # filter self.stock_list by what's in self.all_stock_list
                self.stock_list = [s for s in self.stock_list if s in self.all_stock_list]
                self._filter = self._date_filter & self._sym_filter
            else:
                self._filter = self._date_filter
        else:
            if self.stock_list and symbol_field_idx is not None:
                self._sym_filter = ds.field(symbol_field_idx).isin(self.stock_list)
                self._filter = self._sym_filter
            else:
                self._filter = None

    
    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def to_pandas_dict_fast(self, start_pct: float = 0.0, end_pct: float = 1.0, columns: Optional[List[str]] = None):
        """
        Return a dictionary of pandas DataFrames, one per symbol.
        Optimized version that loads all data at once.
        @param start_pct: *Optional.* Fraction *f* (0 ≤ f ≤ 1) shifting the start, default=0.0
        @param end_pct: *Optional.* Fraction *g* (0 ≤ g ≤ 1) limiting the span, default=1.0
        @param columns: *Optional.* List of column names to include.
        @return: Dictionary of :class:`DataFrame` keyed by symbol.
        """
        if self.ds_parquet is None:
            raise RuntimeError("mkt_data_mana:to_pandas_dict: Dataset not loaded.")

        # filter desired results
        self._compute_effective_bounds(start_pct, end_pct)

        # Determine which symbols to load
        symbols_to_load = self.stock_list if self.stock_list else self.all_stock_list
    
        # Build a single filter for all symbols at once
        schema = self.ds_parquet.schema
        symbol_field_idx = None
        date_field_idx = None
    
        for i, field in enumerate(schema):
            if field.name == "symbol" and symbol_field_idx is None:
                symbol_field_idx = i
            elif field.name == "date" and date_field_idx is None:
                date_field_idx = i
    
        # Create combined filter
        if symbol_field_idx is not None:
            symbol_filter = ds.field(symbol_field_idx).isin(symbols_to_load)
        else:
            symbol_filter = None
    
        # Combine with date filter if available
        if self._date_filter is not None and symbol_filter is not None:
            combined_filter = symbol_filter & self._date_filter
        elif self._date_filter is not None:
            combined_filter = self._date_filter
        elif symbol_filter is not None:
            combined_filter = symbol_filter
        else:
            combined_filter = None
    
        # Load ALL data at once
        table = self.ds_parquet.to_table(
            filter=combined_filter,
            columns=columns
        )
    
        if table.num_rows == 0:
            return {}
    
        # Convert to pandas once
        df_all = table.to_pandas()
    
        # Sort by symbol and date once
        sort_cols = ['symbol', 'date'] if 'date' in df_all.columns else ['symbol']
        df_all = df_all.sort_values(sort_cols)
    
        # Split into dictionary by symbol using groupby
        data_dict = {}
        num_dates_dict = {}
        sym_count = 0

        for symbol, group_df in df_all.groupby('symbol', sort=False):
            # Drop the symbol column and create a copy
            symbol_df = group_df.drop('symbol', axis=1).copy()
            sym_count += 1
        
            # Set date as index if available
            if 'date' in symbol_df.columns:
                symbol_df.set_index('date', inplace=True)
                num_dates_dict[symbol] = symbol_df.index.nunique()
            else:
                symbol_df.reset_index(drop=True, inplace=True)
                num_dates_dict[symbol] = symbol_df.index.nunique()
        
            data_dict[symbol] = symbol_df
    
        # Calculate num_dates
        if num_dates_dict:
            date_counts = list(num_dates_dict.values())
            if sym_count >= 3:
                try:
                    self.num_dates = stats.mode(date_counts)[0]
                except:
                    self.num_dates = int(np.round(np.mean(date_counts)))
            else:
                self.num_dates = int(np.round(np.mean(date_counts)))
        else:
            self.num_dates = 0
    
        logger.info(f"start_date: {self.start_date}, end_date: {self.end_date}, num_dates: {self.num_dates}")
        logger.info(f"Loaded data for {len(data_dict)} symbols: {list(data_dict.keys())[:10]}{'...' if len(data_dict) > 10 else ''}")
    
        return data_dict



    @property
    def all_stock_list(self) -> List[str]:
        """
        Return a cached list of all unique symbols found in the dataset.
        @return: List of unique symbols.
        """
        if self.ds_parquet is None:
            # Ensure the dataset is loaded before accessing symbols
            self.load_bars_from_mana()

        if self._all_stock_list is None:
            # Get unique symbols from partitions or scan
            symbol_set = set()
            
            # First try to get from file paths (faster)
            for rel in self.ds_parquet.files:
                for part in Path(rel).parts:
                    if part.startswith("symbol="):
                        symbol_set.add(part.split("=", 1)[1])
                        break
            
            # If no symbols found in paths, scan the data
            if not symbol_set:
                scanner = self.ds_parquet.scanner(columns=['symbol'])
                for batch in scanner.to_batches():
                    col = batch.column(0)
                    for val in col:
                        if val.is_valid:
                            symbol_set.add(val.as_py())
            
            self._all_stock_list = sorted(list(symbol_set))

        return self._all_stock_list