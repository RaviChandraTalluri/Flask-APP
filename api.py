from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from functools import wraps
from datetime import datetime
import secrets
import logging
from logging.handlers import RotatingFileHandler
import json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
from io import StringIO

app = Flask(__name__, static_folder='static')
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": False
    }
})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = secrets.token_hex(32)  # Generate a secure secret key

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')
    
handler = RotatingFileHandler('logs/api.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info('API startup')

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src * 'unsafe-inline' 'unsafe-eval'; img-src * data:;"
    
    # Ensure CORS headers are set
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    return response

# Update Redis configuration with fallback to in-memory storage
REDIS_URL = os.getenv('REDIS_URL', None)
try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL)
        storage_uri = REDIS_URL
    else:
        storage_uri = "memory://"
        redis_client = None
except:
    storage_uri = "memory://"
    redis_client = None

# Update the limiter configuration to use memory storage if Redis is unavailable
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=storage_uri,
    default_limits=["200 per day", "50 per hour"]
)

# Add error handler for rate limit exceeded
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description)
    }), 429

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def load_data(file_path):
    """Load dataset from file, automatically detecting delimiter and headers."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except Exception as e:
        return None

def clean_data(df):
    """Handle missing values and drop irrelevant columns dynamically."""
    df_cleaned = df.copy()
    df_cleaned.drop_duplicates(inplace=True)
    df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)
    df_cleaned.dropna(axis=1, how='all', inplace=True)
    return df_cleaned

def create_boxplot(data):
    plt.figure(figsize=(8, 4))
    plt.boxplot(data)
    return plt.gcf()

def create_heatmap(corr_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center')
    return plt.gcf()

@app.route('/', methods=['GET'])
@limiter.limit("10 per minute")
def home():
    return jsonify({
        'status': 'API is running',
        'endpoints': {
            '/': 'Home - This message',
            '/test': 'Test endpoint',
            '/analyze': 'Upload and analyze data (POST request)'
        }
    })

@app.route('/analyze', methods=['GET', 'POST'])
@limiter.limit("30 per hour")
def analyze():
    if request.method == 'GET':
        # Handle GET request
        return jsonify({"message": "GET request received"})
    elif request.method == 'POST':
        # Handle POST request
        data = request.json
        # Process the data
        return jsonify({"message": "POST request received", "data": data})

@app.route('/test', methods=['GET'])
@limiter.limit("10 per minute")
def test():
    return jsonify({'status': 'Server is running'}), 200

@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Check Redis connection only if configured
        redis_status = 'not configured'
        if redis_client:
            try:
                redis_client.ping()
                redis_status = 'ok'
            except:
                redis_status = 'error'
        
        # Check if upload directory exists and is writable
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            return jsonify({'status': 'error', 'message': 'Upload directory missing'}), 500
        if not os.access(app.config['UPLOAD_FOLDER'], os.W_OK):
            return jsonify({'status': 'error', 'message': 'Upload directory not writable'}), 500
            
        # Test matplotlib
        plt.figure()
        plt.close()
        
        return jsonify({
            'status': 'healthy',
            'upload_dir': 'ok',
            'matplotlib': 'ok',
            'redis': redis_status,
            'version': '1.0',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        app.logger.error(f'Health check failed: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test-analysis', methods=['GET'])
def test_analysis():
    # Create sample data
    sample_data = [
        {
            'title': 'Test Analysis',
            'insights': [
                'Dataset contains 100 rows and 5 columns',
                'Sample insight 1',
                'Sample insight 2'
            ],
            'visualization': None
        }
    ]
    return jsonify(sample_data)

@app.route('/test-full-analysis', methods=['GET'])
def test_full_analysis():
    try:
        # Create sample CSV data
        sample_data = """date,value,category
2023-01-01,100,A
2023-02-01,120,B
2023-03-01,110,A
2023-04-01,130,B
2023-05-01,125,A
2023-06-01,140,B
2023-07-01,135,A
2023-08-01,150,B
2023-09-01,145,A
2023-10-01,160,B"""

        # Save sample data to temporary file
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'test_data.csv')
        with open(temp_file, 'w') as f:
            f.write(sample_data)

        try:
            # Load and verify data
            df = load_data(temp_file)
            if df is None:
                return jsonify({'error': 'Failed to load test data'}), 500

            # Process data and generate results
            results = []
            
            # Basic data summary
            df_info = StringIO()
            df.info(buf=df_info)
            summary_stats = df.describe().to_dict()
            missing_values = df.isnull().sum().to_dict()
            
            results.append({
                'title': 'Dataset Overview',
                'insights': [
                    f"Dataset contains {len(df)} rows and {len(df.columns)} columns",
                    f"Column Information:\n{df_info.getvalue()}",
                    f"Missing Values: {json.dumps(missing_values, indent=2)}"
                ]
            })

            # Data Distribution
            plt.figure(figsize=(12, 6))
            df['value'].hist(bins=30, color='blue', alpha=0.7)
            plt.title("Distribution of Values")
            results.append({
                'title': 'Distribution Analysis',
                'visualization': fig_to_base64(plt.gcf()),
                'insights': ["Sample distribution analysis of values"]
            })
            plt.close()

            # Time Series Analysis
            df['date'] = pd.to_datetime(df['date'])
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['value'])
            plt.title("Time Series Plot")
            plt.xticks(rotation=45)
            results.append({
                'title': 'Time Series Analysis',
                'visualization': fig_to_base64(plt.gcf()),
                'insights': ["Sample time series analysis"]
            })
            plt.close()

            # Classification Analysis
            label_enc = LabelEncoder()
            df['category_encoded'] = label_enc.fit_transform(df['category'])
            features = df[['value']].values
            target = df['category_encoded'].values
            
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results.append({
                'title': 'Classification Analysis',
                'insights': [
                    f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}",
                    "Sample classification analysis"
                ]
            })

            return jsonify({
                'status': 'success',
                'message': 'Test analysis completed successfully',
                'results': results
            })

        except Exception as e:
            app.logger.error(f'Error in test analysis: {str(e)}', exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Analysis error: {str(e)}'
            }), 500

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)

    except Exception as e:
        app.logger.error(f'Test analysis failed: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/verify-analysis', methods=['GET'])
def verify_analysis():
    try:
        # Test data loading
        sample_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        # Test plotting
        plt.figure()
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.close()
        
        # Test machine learning
        model = RandomForestClassifier(n_estimators=10)
        X = [[1], [2], [3]]
        y = [0, 1, 0]
        model.fit(X, y)
        
        return jsonify({
            'status': 'success',
            'message': 'All analysis components are working',
            'components_tested': {
                'pandas': 'ok',
                'matplotlib': 'ok',
                'scikit-learn': 'ok'
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return a no-content response

if __name__ == '__main__':
    app.logger.info('Starting Flask server...')
    app.run(host='0.0.0.0', port=5000, debug=True) 
