import os
import pandas as pd
from datetime import datetime
from pathlib import Path


# Class-based approach for better organization
class TCADataManager:
    """
    Manages TCA data storage and retrieval.
    Data is stored in a parquet file with the following columns:
    - order_id
    - order_type
    - order_size
    - order_price
    - order_time
    - order_status
    - order_filled_size
    - order_filled_price
    - order_filled_time
    - order_filled_status
    - order_filled_price
    - order_filled_time
    """
    
    def __init__(self, base_dir="tca_data"):
        """
        Initialize the TCA data manager.

        Args:
            base_dir: The directory to store the TCA data.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_orders(self, test_orders, run_id=None):
        """
        Save test orders with optional run identifier.
        
        Args:
            test_orders: A list of test orders.
            run_id: An optional run identifier.

        Returns:
            The file path of the saved data.
        """
        try:
            df = pd.DataFrame(test_orders)
            
            if df.empty:
                print("Warning: No data to save")
                return None
            
            # Include run_id in filename if provided
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if run_id:
                file_name = f"tca_data_{run_id}_{timestamp}.parquet"
            else:
                file_name = f"tca_data_{timestamp}.parquet"
            
            file_path = self.base_dir / file_name
            
            # Save with optimal settings
            df.to_parquet(
                file_path,
                compression='snappy',
                index=False,
                engine='pyarrow'
            )
            
            print(f"Saved {len(df)} orders to: {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return None
    
    def list_files(self, pattern="tca_data_*.parquet"):
        """
        List all TCA parquet files.

        Args:
            pattern: The pattern to match the TCA data files.

        Returns:
            A list of file paths.
        """
        return list(self.base_dir.glob(pattern))
    
    def load_latest(self):
        """
        Load the most recent parquet file.
        
        Returns:
            A pandas DataFrame containing the TCA data.
        """
        files = self.list_files()
        if not files:
            return None
        
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return pd.read_parquet(latest_file)
    
    def cleanup_old_files(self, keep_last_n=10):
        """Keep only the N most recent files."""
        files = sorted(self.list_files(), key=lambda f: f.stat().st_mtime)
        
        if len(files) > keep_last_n:
            for old_file in files[:-keep_last_n]:
                old_file.unlink()
                print(f"Deleted old file: {old_file}")

# Usage examples:

# Simple function approach
# file_path = save_test_orders_to_parquet(test_orders)

# Class-based approach
# data_manager = TCADataManager()
# data_manager.save_orders(test_orders, run_id="backtest_001")
# data_manager.cleanup_old_files(keep_last_n=5)