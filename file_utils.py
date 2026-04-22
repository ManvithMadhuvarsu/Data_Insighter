import pandas as pd
import json
import chardet

def read_data_file(filepath):
    """Read either CSV or JSON file and return a pandas DataFrame"""
    file_extension = filepath.rsplit('.', 1)[1].lower()
    
    try:
        if file_extension == 'csv':
            # First, try to detect the file encoding
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                detected_encoding = result.get('encoding') or 'utf-8'
            
            # Try different approaches to read the CSV
            try:
                # Approach 1: Try with detected encoding and auto-detected separator
                df = pd.read_csv(
                    filepath,
                    encoding=detected_encoding,
                    sep=None,  # Auto-detect separator
                    engine='python',
                    on_bad_lines='skip',
                    quotechar='"',
                    skipinitialspace=True,
                    thousands=',',  # Handle numbers with commas
                    decimal='.'     # Handle decimal points
                )
                if not df.empty:
                    return df
            except:
                pass
            
            try:
                # Approach 2: Try with common encodings and separators
                encodings = [detected_encoding, 'utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']
                separators = [',', ';', '\t', '|']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(
                                filepath,
                                encoding=encoding,
                                sep=sep,
                                engine='python',
                                on_bad_lines='skip',
                                quotechar='"',
                                skipinitialspace=True,
                                thousands=',',
                                decimal='.'
                            )
                            if not df.empty:
                                return df
                        except:
                            continue
            except:
                pass
            
            try:
                # Approach 3: Try reading with default parameters and error handling
                df = pd.read_csv(
                    filepath,
                    on_bad_lines='skip',
                    thousands=',',
                    decimal='.'
                )
                if not df.empty:
                    return df
            except:
                pass
            
            raise ValueError("Could not read CSV file with any supported encoding or separator")
            
        elif file_extension == 'json':
            # First, try to detect the file encoding
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                detected_encoding = result.get('encoding') or 'utf-8'
            
            try:
                # Approach 1: Try reading as regular JSON
                with open(filepath, 'r', encoding=detected_encoding) as f:
                    json_data = json.load(f)
                
                if isinstance(json_data, list):
                    # If it's a list of records
                    df = pd.json_normalize(json_data)
                elif isinstance(json_data, dict):
                    # If it's a single record or nested structure
                    df = pd.json_normalize([json_data])
                else:
                    raise ValueError("Unsupported JSON structure")
                
                if not df.empty:
                    return df
            except:
                pass
            
            try:
                # Approach 2: Try reading as JSON Lines
                df = pd.read_json(filepath, lines=True)
                if not df.empty:
                    return df
            except:
                pass
            
            try:
                # Approach 3: Try reading with different encodings
                encodings = [detected_encoding, 'utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']
                for encoding in encodings:
                    try:
                        with open(filepath, 'r', encoding=encoding) as f:
                            json_data = json.load(f)
                        if isinstance(json_data, list):
                            df = pd.json_normalize(json_data)
                        elif isinstance(json_data, dict):
                            df = pd.json_normalize([json_data])
                        if not df.empty:
                            return df
                    except:
                        continue
            except:
                pass
            
            raise ValueError("Could not read JSON file with any supported encoding")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}") 
