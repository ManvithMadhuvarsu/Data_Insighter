import pandas as pd
import json
import numpy as np
from typing import Dict, List, Any

class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from file with enhanced format support and error handling"""
        file_extension = self.filepath.rsplit('.', 1)[1].lower()
        
        try:
            if file_extension == 'csv':
                # Try different encodings and separators
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']
                separators = [',', ';', '\t', '|']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            # First try with C engine (faster)
                            try:
                                df = pd.read_csv(
                                    self.filepath,
                                    encoding=encoding,
                                    sep=sep,
                                    engine='c',
                                    on_bad_lines='skip'
                                )
                                if not df.empty:
                                    return df
                            except:
                                # If C engine fails, try Python engine
                                df = pd.read_csv(
                                    self.filepath,
                                    encoding=encoding,
                                    sep=sep,
                                    engine='python',
                                    on_bad_lines='skip'
                                )
                                if not df.empty:
                                    return df
                        except UnicodeDecodeError:
                            continue
                        except pd.errors.EmptyDataError:
                            raise ValueError("The CSV file is empty")
                        except Exception as e:
                            print(f"Warning: Failed to read CSV with encoding {encoding} and separator {sep}: {str(e)}")
                            continue
                
                raise ValueError("Could not read CSV file with any supported encoding or separator")
                
            elif file_extension == 'json':
                try:
                    # Try reading as regular JSON
                    with open(self.filepath, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    if isinstance(json_data, list):
                        # If it's a list of records
                        return pd.json_normalize(json_data)
                    elif isinstance(json_data, dict):
                        # If it's a single record or nested structure
                        return pd.json_normalize([json_data])
                    else:
                        raise ValueError("Unsupported JSON structure")
                except json.JSONDecodeError:
                    try:
                        # Try reading as JSON Lines
                        return pd.read_json(self.filepath, lines=True)
                    except:
                        # Try reading with different encodings
                        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']
                        for encoding in encodings:
                            try:
                                with open(self.filepath, 'r', encoding=encoding) as f:
                                    json_data = json.load(f)
                                if isinstance(json_data, list):
                                    return pd.json_normalize(json_data)
                                elif isinstance(json_data, dict):
                                    return pd.json_normalize([json_data])
                            except:
                                continue
                        raise ValueError("Could not read JSON file with any supported encoding")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def _flatten_json(self, nested_json: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested JSON structure"""
        items = []
        for key, value in nested_json.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_json(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    for i, item in enumerate(value):
                        items.extend(self._flatten_json(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((new_key, value))
            else:
                items.append((new_key, value))
        return dict(items)
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data"""
        return {
            'columns': self.df.columns.tolist(),
            'num_rows': len(self.df),
            'num_columns': len(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'preview': self.df.head(5).to_dict(orient='records'),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist()
        }
    
    def sample_data(self, percentage: int) -> pd.DataFrame:
        """Sample the data based on percentage"""
        if percentage >= 100:
            return self.df
        
        sample_size = int(len(self.df) * (percentage / 100))
        return self.df.sample(n=sample_size, random_state=42)
    
    def _get_summary_stats(self) -> Dict[str, Dict[str, Any]]:
        """Calculate summary statistics for each column."""
        summary = {}
        
        for column in self.df.columns:
            col_stats = {}
            
            if pd.api.types.is_numeric_dtype(self.df[column]):
                col_stats.update({
                    'mean': float(self.df[column].mean()),
                    'median': float(self.df[column].median()),
                    'std': float(self.df[column].std()),
                    'min': float(self.df[column].min()),
                    'max': float(self.df[column].max())
                })
            else:
                value_counts = self.df[column].value_counts()
                col_stats.update({
                    'unique_values': len(value_counts),
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'most_common_count': int(value_counts.iloc[0]) if not value_counts.empty else 0
                })
            
            summary[column] = col_stats
        
        return summary
    
    def process_columns(self, columns: List[str]) -> Dict[str, Any]:
        """Process selected columns for analysis."""
        if not all(col in self.df.columns for col in columns):
            raise ValueError("One or more selected columns not found in dataset")
        
        processed_data = {
            'data': self.df[columns].to_dict('records'),
            'correlations': self._get_correlations(columns),
            'quality_metrics': self._get_quality_metrics(columns)
        }
        return processed_data
    
    def _get_correlations(self, columns: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between numeric columns."""
        numeric_cols = [col for col in columns 
                       if pd.api.types.is_numeric_dtype(self.df[col])]
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = self.df[numeric_cols].corr()
        return corr_matrix.to_dict()
    
    def _get_quality_metrics(self, columns: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate data quality metrics for selected columns."""
        metrics = {}
        
        for column in columns:
            col_metrics = {
                'completeness': 1 - (self.df[column].isnull().sum() / len(self.df)),
                'uniqueness': len(self.df[column].unique()) / len(self.df)
            }
            
            if pd.api.types.is_numeric_dtype(self.df[column]):
                col_metrics['zeros_percentage'] = (self.df[column] == 0).sum() / len(self.df)
                col_metrics['negative_percentage'] = (self.df[column] < 0).sum() / len(self.df)
            
            metrics[column] = col_metrics
        
        return metrics 