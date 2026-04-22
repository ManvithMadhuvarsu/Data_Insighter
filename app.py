from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import json
from functools import wraps
from data_processor import DataProcessor
from visualization_generator import VisualizationGenerator
import numpy as np
from datetime import datetime
import time
import secrets
from file_utils import read_data_file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# FIX #1: Secret key from environment variable, never hardcoded
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['SAMPLE_DATASETS'] = 'sample_datasets'

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLE_DATASETS'], exist_ok=True)
os.makedirs(os.path.join('static', 'css'), exist_ok=True)
os.makedirs(os.path.join('static', 'js'), exist_ok=True)
os.makedirs(os.path.join('static', 'images'), exist_ok=True)

# ---------- User store helpers (FIX #2 + #3) ----------
USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def _load_users():
    """Load users from JSON file."""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def _save_users(users):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)

# Start with an empty user store; accounts should be created through registration.
if not os.path.exists(USERS_FILE):
    _save_users({})

# ---------- CSRF helpers (FIX #4) ----------
def generate_csrf_token():
    """Generate a CSRF token and store it in the session."""
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']

def validate_csrf_token():
    """Validate the CSRF token from form data or JSON against the session."""
    payload = request.get_json(silent=True) or {}
    token = request.form.get('_csrf_token') or payload.get('_csrf_token', '')
    if not token or token != session.get('_csrf_token'):
        return False
    return True

# Make csrf_token available in all templates
app.jinja_env.globals['csrf_token'] = generate_csrf_token

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'json'}

def error_response(message, status_code):
    return jsonify({'success': False, 'error': message}), status_code

def is_path_within_directory(directory, target_path):
    """Ensure target_path resolves inside directory."""
    try:
        base_dir = os.path.abspath(directory)
        target = os.path.abspath(target_path)
        return os.path.commonpath([base_dir, target]) == base_dir
    except ValueError:
        return False

def cleanup_uploaded_file(filepath):
    """Delete files only when they are inside the uploads directory."""
    if filepath and os.path.exists(filepath) and is_path_within_directory(app.config['UPLOAD_FOLDER'], filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

def login_required(view):
    """Require an authenticated session for app pages and APIs."""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if session.get('user'):
            return view(*args, **kwargs)

        if request.is_json or request.method != 'GET':
            return error_response('Authentication required', 401)

        flash('Please log in to continue.', 'error')
        return redirect(url_for('login'))
    return wrapped_view

def flatten_json_data(data):
    """Recursively flatten nested JSON structures"""
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if len(v) > 0:
                    if isinstance(v[0], dict):
                        # Handle list of dictionaries
                        for i, item in enumerate(v):
                            items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        # Handle list of simple values
                        items.append((new_key, v))
                else:
                    items.append((new_key, v))
            else:
                items.append((new_key, v))
        return dict(items)

    if isinstance(data, list):
        return [flatten_dict(item) if isinstance(item, dict) else item for item in data]
    elif isinstance(data, dict):
        return flatten_dict(data)
    else:
        return data

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

@app.route('/')
@login_required
def index():
    # FIX #6: Only clear file-related keys, NEVER clear user/login state
    if 'current_filepath' in session:
        cleanup_uploaded_file(session.get('current_filepath'))
        session.pop('current_filepath', None)
    
    sample_datasets = [f for f in os.listdir(app.config['SAMPLE_DATASETS']) 
                      if allowed_file(f)]
    return render_template('index.html', sample_datasets=sample_datasets)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if not validate_csrf_token():
            return error_response('Invalid form submission. Please refresh and try again.', 400)

        if 'file' not in request.files:
            return error_response('No file part', 400)
        
        file = request.files['file']
        if file.filename == '':
            return error_response('No selected file', 400)
        
        if file and allowed_file(file.filename):
            filepath = None
            try:
                # Clear any existing uploaded file
                if 'current_filepath' in session:
                    cleanup_uploaded_file(session.get('current_filepath'))
                
                # Create a safe filename
                original_name = secure_filename(file.filename)
                extension = file.filename.rsplit('.', 1)[1].lower()
                basename = os.path.splitext(original_name)[0] or 'upload'
                filename = f"{basename}_{int(time.time())}_{secrets.token_hex(4)}.{extension}"
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Store filepath in session
                session['current_filepath'] = filepath
                
                # Read the file
                df = read_data_file(filepath)
                
                # Prepare preview data
                preview_data = []
                for _, row in df.head(10).iterrows():
                    row_dict = {}
                    for col in df.columns:
                        value = row[col]
                        if isinstance(value, (np.integer, np.floating)):
                            value = float(value)
                        elif isinstance(value, np.bool_):
                            value = bool(value)
                        elif isinstance(value, (pd.Timestamp, datetime)):
                            value = value.strftime('%Y-%m-%d %H:%M:%S')
                        elif isinstance(value, (list, dict)):
                            value = json.dumps(value)
                        elif pd.isna(value):
                            value = None
                        else:
                            value = str(value)
                        row_dict[col] = value
                    preview_data.append(row_dict)

                # Prepare column information
                columns_info = []
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    non_null_count = df[col].count()
                    null_count = df[col].isna().sum()
                    unique_count = df[col].nunique()
                    
                    columns_info.append({
                        'name': col,
                        'type': col_type,
                        'non_null_count': int(non_null_count),
                        'null_count': int(null_count),
                        'unique_count': int(unique_count)
                    })

                data_info = {
                    'filename': filename,
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'columns': columns_info,
                    'preview_data': preview_data
                }
                
                return jsonify({
                    'success': True,
                    'data': data_info
                })
                
            except Exception as e:
                # Clean up the file if there was an error
                cleanup_uploaded_file(filepath)
                return error_response(str(e), 400)

        return error_response('Unsupported file format. Please upload a CSV or JSON file.', 400)
    
    # Get list of sample datasets for the template
    sample_datasets = [f for f in os.listdir(app.config['SAMPLE_DATASETS']) 
                      if allowed_file(f)]
    return render_template('upload.html', sample_datasets=sample_datasets)

@app.route('/sample/<filename>')
@login_required
def use_sample(filename):
    filepath = os.path.join(app.config['SAMPLE_DATASETS'], secure_filename(filename))
    # FIX #5: Validate the path stays inside SAMPLE_DATASETS directory
    if not is_path_within_directory(app.config['SAMPLE_DATASETS'], filepath):
        return error_response('Invalid file path', 400)
    
    if os.path.exists(filepath) and allowed_file(filename):
        try:
            df = read_data_file(filepath)
            # FIX #5: Store filepath only in server session, never send to client
            session['current_filepath'] = filepath
            
            data_info = {
                'columns': df.columns.tolist(),
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'preview': df.head(5).to_dict(orient='records'),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            return jsonify({
                'success': True,
                'data': data_info
            })
        except Exception as e:
            return error_response(str(e), 400)
    return error_response('Sample dataset not found', 404)

@app.route('/analysis')
@login_required
def analysis():
    # FIX #5: Never read filepath from query params — use session only
    filepath = session.get('current_filepath')
    if not filepath or not os.path.exists(filepath):
        flash('Please upload a file or select a sample dataset first', 'error')
        return redirect(url_for('index'))
    try:
        df = read_data_file(filepath)
        columns = df.columns.tolist()
        return render_template('analysis.html', columns=columns)
    except Exception as e:
        flash(f'Error loading data file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analysis_summary')
@login_required
def analysis_summary():
    filepath = session.get('current_filepath')
    if not filepath or not os.path.exists(filepath):
        return error_response('No data file loaded. Please upload a file first.', 400)

    try:
        processor = DataProcessor(filepath)
        return jsonify({
            'success': True,
            'summary': processor.get_analysis_summary()
        })
    except Exception as e:
        return error_response(str(e), 400)

@app.route('/dashboard')
@login_required
def dashboard():
    if not session.get('current_filepath'):
        flash('Please upload a file or select a sample dataset first', 'error')
        return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/generate_visualization', methods=['POST'])
@login_required
def generate_visualization():
    try:
        if not validate_csrf_token():
            return error_response('Invalid request token', 400)

        data = request.json
        # FIX #5: Never accept filepath from client — always use session
        filepath = session.get('current_filepath')
        columns = data.get('columns')
        viz_type = data.get('type')
        sample_percentage = data.get('sample_percentage', 100)
        
        if not filepath:
            return error_response('No data file loaded. Please upload a file first.', 400)
            
        if not columns:
            return error_response('No columns selected', 400)
            
        if not viz_type:
            return error_response('No visualization type selected', 400)
        
        if not os.path.exists(filepath):
            return error_response('Data file not found. Please upload again.', 400)
        
        viz_generator = VisualizationGenerator(filepath)
        visualization = viz_generator.generate_visualization(
            columns, 
            viz_type, 
            sample_percentage
        )
        
        return jsonify({
            'success': True,
            'visualization': visualization
        })
        
    except Exception as e:
        import traceback
        print(f"Error in generate_visualization: {str(e)}")
        print(traceback.format_exc())
        return error_response(str(e), 400)

@app.route('/export', methods=['POST'])
@login_required
def export_visualization():
    try:
        if not validate_csrf_token():
            return error_response('Invalid request token', 400)

        data = request.json
        viz_type = data.get('type')
        viz_data = data.get('data')
        
        if not viz_type or not viz_data:
            return error_response('Missing required parameters', 400)
        
        viz_generator = VisualizationGenerator(None)
        export_file = viz_generator.export_visualization(viz_type, viz_data)
        
        return send_file(
            export_file,
            as_attachment=True,
            download_name=f'visualization.{viz_type}'
        )
    except Exception as e:
        return error_response(str(e), 400)

@app.route('/export_dashboard', methods=['POST'])
@login_required
def export_dashboard():
    try:
        if not validate_csrf_token():
            return error_response('Invalid request token', 400)

        data = request.json
        dashboard_data = data.get('dashboard_data')
        
        if not dashboard_data:
            return error_response('Missing dashboard data', 400)
        
        # Read the template
        with open('templates/saving_dashboard.html', 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Replace the placeholder with actual dashboard data
        dashboard_json = json.dumps(dashboard_data)
        html_content = template_content.replace('DASHBOARD_DATA_PLACEHOLDER', dashboard_json)
        
        # Create a temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'dashboard_export_{timestamp}.html'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)
        
        # Save the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_file,
            mimetype='text/html'
        )
        
    except Exception as e:
        return error_response(str(e), 400)

@app.route('/clear_session', methods=['POST'])
@login_required
def clear_session():
    try:
        if not validate_csrf_token():
            return error_response('Invalid request token', 400)

        # Get the current filepath from session
        current_filepath = session.get('current_filepath')
        
        # If there's a file and it's in the uploads directory, delete it
        cleanup_uploaded_file(current_filepath)

        # Clear all files in the uploads directory that are older than 1 hour
        current_time = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if not is_path_within_directory(app.config['UPLOAD_FOLDER'], filepath):
                    continue
                if os.path.getctime(filepath) < (current_time - 3600):  # 3600 seconds = 1 hour
                    os.remove(filepath)
            except Exception as e:
                print(f"Error removing old file {filename}: {str(e)}")
        
        # FIX #6: Only clear file-related keys, preserve login state
        session.pop('current_filepath', None)
        
        return jsonify({'success': True, 'message': 'Session cleared successfully'})
    except Exception as e:
        return error_response(str(e), 500)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # FIX #4: Validate CSRF token
        if not validate_csrf_token():
            flash('Invalid form submission. Please try again.', 'error')
            return redirect(url_for('register'))
        
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validate inputs
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('register'))
        
        # FIX #3: Actually save the user with a hashed password
        users = _load_users()
        if username in users:
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        users[username] = {
            'email': email,
            'password_hash': generate_password_hash(password)
        }
        _save_users(users)
        
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # FIX #4: Validate CSRF token
        if not validate_csrf_token():
            flash('Invalid form submission. Please try again.', 'error')
            return redirect(url_for('login'))
        
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # FIX #2: Authenticate against stored users with hashed passwords
        users = _load_users()
        matched_username = username
        user = users.get(username)
        if user is None:
            matched = next(
                (
                    (stored_username, details)
                    for stored_username, details in users.items()
                    if details.get('email', '').lower() == username.lower()
                ),
                None
            )
            if matched:
                matched_username, user = matched
        
        if user and check_password_hash(user['password_hash'], password):
            session['user'] = matched_username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Proper logout route."""
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
