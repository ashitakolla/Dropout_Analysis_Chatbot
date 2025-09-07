import os
import hashlib
import json
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, session

# Load environment variables from .env file
load_dotenv()
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import logging
from pathlib import Path
from chatbot import ChatbotService

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webapp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the enhanced student data
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'students_enhanced.csv'
MODEL_PATH = BASE_DIR / 'output' / 'best_model.pkl'

# User class for Flask-Login
class User:
    def __init__(self, username, password_hash, first_name, last_name, is_teacher=False):
        self.username = username
        self.password_hash = password_hash
        self.first_name = first_name
        self.last_name = last_name
        self.is_teacher = is_teacher
        self.is_authenticated = False  # Changed from True to False
        self.is_active = True
        self.is_anonymous = False

    def get_id(self):
        return self.username

    def check_password(self, password):
        # Check if the stored password is already a hash (64 chars)
        if len(self.password_hash) == 64 and all(c in '0123456789abcdef' for c in self.password_hash):
            # Compare with hashed password
            return self.password_hash == hashlib.sha256(password.encode()).hexdigest()
        else:
            # Compare plaintext password
            return self.password_hash == password

# Load user data
def load_users():
    try:
        # Load student credentials
        creds_path = BASE_DIR / 'student_credentials.csv'
        creds_df = pd.read_csv(creds_path)
        
        # Load student data
        student_df = pd.read_csv(DATA_PATH)
        
        users = {}
        
        # Add students from credentials
        for _, row in creds_df.iterrows():
            # Find the student in the main dataset
            student_row = student_df[student_df['username'] == row['username']]
            if not student_row.empty:
                first_name = student_row.iloc[0].get('first_name', row.get('first_name', 'Student'))
                last_name = student_row.iloc[0].get('last_name', row.get('last_name', 'User'))
                
                # Create user with the password from credentials
                users[row['username']] = User(
                    username=row['username'],
                    password_hash=row['password'],  # Using the plaintext password
                    first_name=first_name,
                    last_name=last_name,
                    is_teacher=False
                )
        
        # Add a default teacher account (in production, this would be in a database)
        teacher_password = os.environ.get('TEACHER_PASSWORD', 'teacher123')
        teacher_hash = hashlib.sha256(teacher_password.encode()).hexdigest()
        users['teacher'] = User(
            username='teacher',
            password_hash=teacher_hash,
            first_name='Teacher',
            last_name='Admin',
            is_teacher=True
        )
        
        return users
    except Exception as e:
        logger.error(f"Error loading user data: {e}")
        # Return at least the teacher account if something goes wrong
        teacher_hash = hashlib.sha256('teacher123'.encode()).hexdigest()
        return {
            'teacher': User('teacher', teacher_hash, 'Teacher', 'Admin', True)
        }

# Initialize user data
users_db = load_users()

def load_dropout_model():
    """Load the dropout prediction model with improved error handling."""
    model_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_comparison', 'best_model.pkl'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'best_model.pkl'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'best_model.pkl')
    ]
    
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                logger.info(f"Attempting to load model from: {model_path}")
                model = joblib.load(model_path)
                # Verify the model has the required methods
                if hasattr(model, 'predict_proba') and hasattr(model, 'predict'):
                    logger.info(f"Successfully loaded model from {model_path}")
                    logger.info(f"Model type: {type(model).__name__}")
                    # Log model features if available
                    if hasattr(model, 'feature_importances_'):
                        logger.info(f"Model has {len(model.feature_importances_)} features")
                    return model
                else:
                    logger.warning(f"Model at {model_path} is missing required methods")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}", exc_info=True)
    
    logger.error("No valid model file found with required methods. Using fallback prediction method.")
    return None

# Initialize the model when the application starts
model = load_dropout_model()

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(username):
    return users_db.get(username)

# Routes
@app.route('/debug/users')
def debug_users():
    """Debug route to list all users"""
    users_list = []
    for username, user in users_db.items():
        users_list.append({
            'username': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'is_teacher': user.is_teacher
        })
    return jsonify(users_list)

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.is_teacher:
            return redirect(url_for('teacher_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        logger.info(f"Login attempt - Username: {username}")
        
        # Debug: Print all available usernames
        logger.info(f"Available usernames: {list(users_db.keys())}")
        
        user = users_db.get(username)
        logger.info(f"User found in database: {user is not None}")
        
        if user:
            logger.info(f"Checking password for user: {username}")
            logger.info(f"Stored password hash: {user.password_hash}")
            logger.info(f"Provided password: {password}")
            logger.info(f"Hashed provided password: {hashlib.sha256(password.encode()).hexdigest()}")
            
            if user.check_password(password):
                logger.info(f"Password check passed for user: {username}")
                user.is_authenticated = True
                # Use remember=True to keep the user logged in
                login_user(user, remember=True)
                session.permanent = True  # Make the session permanent
                next_page = request.args.get('next')
                logger.info(f"User {username} logged in successfully")
                if user.is_teacher:
                    return redirect(next_page or url_for('teacher_dashboard'))
                else:
                    return redirect(next_page or url_for('student_dashboard'))
            else:
                logger.warning(f"Invalid password for user: {username}")
                logger.warning(f"Expected: {user.password_hash}, Got: {hashlib.sha256(password.encode()).hexdigest()}")
        else:
            logger.warning(f"User not found: {username}")
            logger.warning(f"Available users: {list(users_db.keys())}")
            
        flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    if current_user.is_teacher:
        return redirect(url_for('teacher_dashboard'))
    
    try:
        # Load student data
        df = pd.read_csv(DATA_PATH)
        
        # Find the student by username
        student_data = df[df['username'].str.lower() == current_user.username.lower()]
        
        # If no student found with that username, try to find by name
        if student_data.empty:
            student_data = df[
                (df['first_name'].str.lower() + ' ' + df['last_name'].str.lower()) == 
                current_user.username.lower().replace('_', ' ').strip()
            ]
        
        if student_data.empty:
            logger.error(f"Student data not found for user: {current_user.username}")
            flash('Student data not found. Please contact support.', 'danger')
            return redirect(url_for('logout'))
            
        student_data = student_data.iloc[0].to_dict()
        
        # Log the student data being used for prediction
        logger.info(f"Predicting for student: {student_data.get('first_name', 'Unknown')} {student_data.get('last_name', '')}")
        logger.info(f"Student data: {json.dumps(student_data, default=str, indent=2)}")
        
        # Generate predictions and explanations
        prediction, confidence, explanations = predict_dropout_risk(student_data)
        what_if_scenarios = generate_what_if_scenarios(student_data)
        
        # Log the prediction results
        logger.info(f"Prediction result - High Risk: {prediction}, Confidence: {confidence}%")
        logger.info(f"Key factors: {[e[0] for e in explanations[:3]]}")
        
        # Get department stats
        department = student_data.get('Department', 'Unknown')
        department_stats = {
            'total_students': len(df[df['Department'] == department]) if 'Department' in df.columns else 0,
            'avg_attendance': df[df['Department'] == department]['Attendance'].str.extract(r'(\d+)')[0].astype(float).mean() 
                             if 'Attendance' in df.columns and not df[df['Department'] == department]['Attendance'].empty else 0,
            'avg_overall': df[df['Department'] == department]['Overall'].mean() 
                          if 'Overall' in df.columns and not df[df['Department'] == department]['Overall'].empty else 0
        }
        
        return render_template(
            'student_dashboard.html',
            student=student_data,
            prediction=prediction,
            confidence=min(99, max(1, confidence)),  # Ensure confidence is between 1-99%
            explanations=explanations[:5],  # Show top 5 factors
            what_if_scenarios=what_if_scenarios,
            department_stats=department_stats
        )
        
    except Exception as e:
        logger.error(f"Error in student_dashboard: {str(e)}", exc_info=True)
        flash('An error occurred while loading your dashboard. The support team has been notified.', 'danger')
        return redirect(url_for('index'))

@app.route('/teacher/dashboard')
@login_required
def teacher_dashboard():
    if not current_user.is_teacher:
        return redirect(url_for('student_dashboard'))
    
    # Load and analyze student data
    df = pd.read_csv(DATA_PATH)
    
    # Calculate dropout risk based on Overall grade (students with grade < 2.0 are considered at risk)
    df['dropout_risk'] = df['Overall'].apply(lambda x: 1 if x < 2.0 else 0)
    
    # Generate statistics
    stats = {
        'total_students': len(df),
        'dropout_risk': df['dropout_risk'].mean() * 100,
        'by_department': df['Department'].value_counts().to_dict(),
        'by_gender': df['Gender'].value_counts().to_dict(),
        'avg_attendance': df['Attendance'].str.extract(r'(\d+)')[0].astype(float).mean(),
        'avg_overall': df['Overall'].mean()
    }
    
    # Generate visualizations
    plots = {
        'attendance_dist': plot_attendance_distribution(df),
        'risk_by_department': plot_risk_by_department(df),
        'performance_vs_risk': plot_performance_vs_risk(df)
    }
    
    return render_template(
        'teacher_dashboard.html',
        stats=stats,
        plots=plots,
        students=df.to_dict('records')[:50]  # Show first 50 students
    )

# Helper functions for predictions and visualizations
def preprocess_student_data(student_data):
    """
    Preprocess student data for model prediction.
    Converts categorical variables to numerical and handles missing values.
    The model expects the following features after one-hot encoding:
    - Department (categorical)
    - Gender (categorical)
    - Hometown (categorical)
    - Computer (categorical)
    - Preparation (categorical, converted to study hours)
    - Gaming (categorical)
    - Job (binary)
    - English (categorical)
    - Extra (binary)
    - Income (categorical)
    - HSC (numerical)
    - SSC (numerical)
    """
    try:
        # Create a copy to avoid modifying the original data
        processed = {}
        
        # 1. Last semester grade (0-4 scale) - Not used in the original model
        try:
            processed['last'] = max(0, min(4, float(student_data.get('Last', 0))))
        except (ValueError, TypeError):
            processed['last'] = 0.0
            
        # 2. Overall GPA (0-4 scale) - Not used in the original model
        try:
            processed['overall'] = max(0, min(4, float(student_data.get('Overall', 0))))
        except (ValueError, TypeError):
            processed['overall'] = 0.0
        
        # Categorical features with default values
        categorical_features = {
            'Department': 'Unknown',
            'Gender': 'Unknown',
            'Hometown': 'Urban',
            'Computer': 'No',
            'Preparation': '0-1 Hour',
            'Gaming': 'No',
            'Job': 'No',
            'English': 'No',
            'Extra': 'No',
            'Income': 'Unknown'
        }
        
        # Update with available data
        for feature in categorical_features:
            processed[feature] = str(student_data.get(feature, categorical_features[feature])).strip()
        
        # Numerical features with default values
        numerical_features = {
            'HSC': 0.0,
            'SSC': 0.0
        }
        
        # Update with available data
        for feature in numerical_features:
            try:
                processed[feature] = float(student_data.get(feature, numerical_features[feature]))
            except (ValueError, TypeError):
                processed[feature] = numerical_features[feature]
        
        # Convert categorical features to one-hot encoded format expected by the model
        # This is a simplified version - in a real app, you'd use the same one-hot encoder as during training
        
        # For now, we'll use the fallback prediction since we don't have all the original features
        logger.warning("Using fallback prediction due to missing features")
        return None
        
    except Exception as e:
        logger.error(f"Error preprocessing student data: {e}", exc_info=True)
        return None

def predict_dropout_risk(student_data):
    """
    Predict dropout risk for a student using the trained model or fallback heuristics.
    Returns: (is_high_risk, confidence, explanations)
    """
    global model
    
    # Validate input data
    if not student_data:
        logger.error("No student data provided for prediction")
        return True, 80.0, [('Error: No student data provided', 1.0)]
    
    # Check for critical risk factors first (override model if needed)
    attendance = str(student_data.get('Attendance', '0%')).strip()
    overall_gpa = float(student_data.get('Overall', 0))
    last_sem_grade = float(student_data.get('Last', 0))
    
    # Critical Risk Check 1: Very low attendance
    try:
        if '%' in attendance:
            attendance_pct = int(attendance.replace('%', '').strip())
            if attendance_pct < 40:
                return True, 95.0, [
                    ('Critical: Very Low Attendance (Below 40%)', 1.0),
                    ('High Risk: Low GPA (1.5/4.0)', 1.0) if overall_gpa <= 1.5 else ('', 0),
                    ('Warning: Low Last Semester Grade', 0.8) if last_sem_grade < 2.0 else ('', 0)
                ]
    except (ValueError, AttributeError) as e:
        logger.warning(f"Error parsing attendance data: {e}")
    
    # Try to use the model if available
    try:
        if model is None:
            model = load_dropout_model()
        
        # Preprocess the student data
        features = preprocess_student_data(student_data)
        
        # If preprocessing failed or returned None, use fallback
        if features is None:
            logger.info("Using fallback prediction due to missing features")
            return fallback_heuristic_prediction(student_data)
            
        if model is not None:
            logger.info("Using model for prediction")
            try:
                # Get predictions
                proba = model.predict_proba([features])[0]
                dropout_prob = proba[1]  # Probability of class 1 (dropout)
                prediction = dropout_prob > 0.5
                confidence = max(proba) * 100
                
                # Generate explanations
                explanations = get_feature_explanations(model, features)
                return prediction, confidence, explanations
            except Exception as e:
                logger.error(f"Error in model prediction: {e}", exc_info=True)
                # Fall back to heuristic prediction on error
                return fallback_heuristic_prediction(student_data)
        
        # Fall back to heuristic prediction if model is not available
        logger.info("No model available, using heuristic prediction")
        return fallback_heuristic_prediction(student_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_dropout_risk: {e}", exc_info=True)
        # Fall back to heuristic prediction on error
        return fallback_heuristic_prediction(student_data)

def fallback_heuristic_prediction(student_data):
    """
    Fallback prediction logic when the model is not available.
    Uses a weighted scoring system based on key risk factors.
    """
    try:
        risk_score = 0
        factors = []
        
        # 1. Attendance (0-50 points)
        attendance = str(student_data.get('Attendance', '0%')).strip()
        attendance_pct = 0
        
        # Parse attendance percentage
        try:
            if 'below' in attendance.lower():
                attendance_pct = float(attendance.lower().replace('below', '').replace('%', '').strip()) - 1
            elif '%' in attendance:
                attendance_pct = float(attendance.replace('%', '').strip())
        except (ValueError, AttributeError):
            pass
            
        # Calculate attendance risk
        if attendance_pct < 40:
            risk_score += 50
            factors.append(('Critical: Very Low Attendance (Below 40%)', 0.5))
        elif attendance_pct < 60:
            risk_score += 30
            factors.append(('High: Low Attendance (40-59%)', 0.3))
        elif attendance_pct < 75:
            risk_score += 15
            factors.append(('Moderate: Below Average Attendance (60-74%)', 0.15))
        elif attendance_pct >= 90:
            risk_score -= 10  # Bonus for good attendance
            factors.append(('Positive: Excellent Attendance (90%+)', -0.1))
        
        # 2. GPA (0-50 points)
        gpa = float(student_data.get('Overall', 0))
        if gpa < 1.0:
            risk_score += 50
            factors.append(('Critical: Extremely Low GPA (Below 1.0)', 0.5))
        elif gpa < 1.5:
            risk_score += 40
            factors.append(('High: Very Low GPA (1.0-1.4)', 0.4))
        elif gpa < 2.0:
            risk_score += 30
            factors.append(('Moderate: Low GPA (1.5-1.9)', 0.3))
        elif gpa < 2.5:
            risk_score += 15
            factors.append(('Moderate: Below Average GPA (2.0-2.4)', 0.15))
        elif gpa >= 3.5:
            risk_score -= 15  # Bonus for good GPA
            factors.append(('Positive: High GPA (3.5+)', -0.15))
        
        # 3. Study Time (0-20 points)
        study_time = str(student_data.get('Preparation', '0-1 Hour')).lower()
        if '0-1' in study_time or 'less than 1' in study_time:
            risk_score += 20
            factors.append(('High: Minimal Study Time (0-1 hour)', 0.2))
            # Additional risk for low study time + low GPA
            if gpa < 2.0:
                risk_score += 10
                factors.append(('High: Low Study Time with Low GPA', 0.1))
        elif '2-3' in study_time:
            risk_score += 5
            factors.append(('Moderate: Average Study Time (2-3 hours)', 0.05))
        else:  # More than 3 hours
            risk_score -= 10  # Bonus for good study habits
            factors.append(('Positive: Strong Study Habits (3+ hours)', -0.1))
        
        # 4. Job Status (0-15 points)
        job_status = str(student_data.get('Job', 'no')).lower().strip()
        if job_status in ('yes', 'y', '1', 'true'):
            risk_score += 15
            factors.append(('Moderate: Has Job (increases time pressure)', 0.15))
        
        # 5. Extracurricular Activities (0-10 points)
        extra = str(student_data.get('Extra', 'no')).lower().strip()
        if extra in ('yes', 'y', '1', 'true'):
            risk_score -= 10  # Bonus for being involved
            factors.append(('Positive: Participates in Extracurriculars', -0.1))
        
        # Calculate final risk (0-100)
        risk_score = max(0, min(100, risk_score))
        
        # Determine risk level (threshold at 40% for early years, 50% otherwise)
        semester = str(student_data.get('Semester', '1st')).lower()
        threshold = 40 if any(x in semester for x in ['1st', '2nd', 'first', 'second']) else 50
        
        is_high_risk = risk_score >= threshold
        
        # Calculate confidence based on distance from threshold
        distance = abs(risk_score - threshold)
        confidence = min(95, max(60, 60 + (35 * (distance / (100 - threshold)))))
        
        # Sort factors by absolute weight (descending)
        factors = sorted([f for f in factors if f[0]], key=lambda x: abs(x[1]), reverse=True)
        
        # Ensure we have at least one factor
        if not factors:
            factors = [('Insufficient data for detailed analysis', 0.5)]
        
        return is_high_risk, confidence, factors[:5]  # Return top 5 factors
        
    except Exception as e:
        logger.error(f"Error in fallback prediction: {e}", exc_info=True)
        # Default to high risk if something goes wrong
        return True, 80.0, [('System Error: Using default risk assessment', 1.0)]
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}", exc_info=True)
        # Don't return yet, fall through to heuristic prediction
    
    # Fallback to enhanced heuristic-based prediction if model fails or is not available
    try:
        risk_score = 0
        factors = []
        
        # Critical Risk Factors (automatic high risk if any are true)
        attendance = str(student_data.get('Attendance', '0%')).strip()
        overall_gpa = float(student_data.get('Overall', 0))
        last_sem_grade = float(student_data.get('Last', 0))
        
        # Check for automatic high-risk conditions - handle different attendance formats
        attendance_pct = None
        if 'below' in attendance.lower():
            # Handle 'Below X%' format
            try:
                attendance_pct = float(attendance.lower().replace('below', '').replace('%', '').strip())
                if attendance_pct < 40:
                    attendance_pct = 39.0  # Treat any 'Below X%' where X < 40 as 39%
            except (ValueError, AttributeError):
                pass
        elif '%' in attendance:
            # Handle 'X%' format
            try:
                attendance_pct = float(attendance.replace('%', '').strip())
            except (ValueError, AttributeError):
                pass
                
        if attendance_pct is not None and attendance_pct < 40:
            return True, 95.0, [
                ('Critical: Very Low Attendance (Below 40%)', 1.0),
                ('High Risk: Extremely Low GPA (1.5/4.0)', 1.0) if overall_gpa <= 1.5 else ('', 0),
                ('Warning: Low Last Semester Grade', 0.8) if last_sem_grade < 2.0 else ('', 0)
            ]
            
        if overall_gpa <= 1.0:  # Extremely low GPA
            return True, 90.0, [
                ('Critical: Extremely Low GPA (1.0 or below)', 1.0),
                ('Attendance: ' + attendance, 0.8) if attendance else ('', 0),
                ('Last Semester Grade: ' + str(last_sem_grade), 0.7) if last_sem_grade < 2.0 else ('', 0)
            ]
        
        # Attendance (0-50 points) - Increased weight for attendance
        if '%' in attendance:
            attendance_pct = int(attendance.replace('%', '').strip())
            if attendance_pct < 40:
                risk_score += 50
                factors.append(('Critical: Very Low Attendance (Below 40%)', 0.5))
            elif attendance_pct < 60:
                risk_score += 30
                factors.append(('High: Low Attendance (40-59%)', 0.3))
            elif attendance_pct < 75:
                risk_score += 15
                factors.append(('Moderate: Below Average Attendance (60-74%)', 0.15))
            elif attendance_pct >= 90:
                risk_score -= 10  # Bonus for good attendance
                factors.append(('Positive: Excellent Attendance (90%+)', -0.1))
        elif '%' in attendance:
            pct = int(attendance.replace('%', '').split('-')[0])
            if pct < 50:  # More severe threshold
                risk_score += 40
                factors.append(('High Risk: Very Low Attendance (Below 50%)', 0.4))
            elif pct < 65:
                risk_score += 30
                factors.append(('Moderate Risk: Low Attendance (50-65%)', 0.3))
            elif pct < 80:
                risk_score += 10
                factors.append(('Mild Risk: Moderate Attendance (65-80%)', 0.1))
            # Add positive factor for high attendance
            elif pct >= 90:  # More achievable threshold for positive reinforcement
                risk_score -= 15  # Increased positive impact
                factors.append(('Excellent Attendance (90%+)', -0.15))
        
        # GPA (0-50 points) - Increased weight for GPA
        gpa = float(student_data.get('Overall', 0))
        if gpa < 1.0:
            risk_score += 60  # Critical risk factor
            factors.append(('Critical: Very Low GPA (Below 1.0)', 0.6))
        elif gpa < 1.5:
            risk_score += 50
            factors.append(('Critical: Extremely Low GPA (1.0-1.5)', 0.5))
        elif gpa < 2.0:
            risk_score += 40
            factors.append(('High: Low GPA (1.5-1.9)', 0.4))
        elif gpa < 2.5:
            risk_score += 20
            factors.append(('Moderate: Below Average GPA (2.0-2.4)', 0.2))
        elif gpa >= 3.5:
            risk_score -= 20  # Increased bonus for good GPA
            factors.append(('Positive: High GPA (3.5+)', -0.2)) # Excellent performance
            risk_score -= 25  # Increased positive impact
            factors.append(('Excellent: High GPA (3.5+)', -0.25))
        elif gpa >= 3.0:  # Good performance
            risk_score -= 15
            factors.append(('Good: Above Average GPA (3.0-3.5)', -0.15))
        
        # Study Time (0-30 points)
        study_time = str(student_data.get('Preparation', '')).lower()
        if 'more than 3' in study_time:
            risk_score -= 15  # Bonus for good study habits
            factors.append(('Positive: Excellent Study Habits (3+ hours)', -0.15))
        elif '2-3' in study_time:
            risk_score -= 5
            factors.append(('Positive: Good Study Habits (2-3 hours)', -0.05))
        elif '0-1' in study_time:
            risk_score += 20
            factors.append(('High: Minimal Study Time (0-1 hour)', 0.2))
            
            # Additional risk for low study time + low GPA
            if gpa < 2.0:
                risk_score += 15
                factors.append(('High: Low Study Time with Low GPA', 0.15))
        
        # Final risk calculation with adjusted thresholds
        risk_score = max(0, min(100, risk_score))  # Clamp between 0-100
        
        # Adjust threshold based on academic year
        semester = str(student_data.get('Semester', '1st')).lower()
        if '1st' in semester or '2nd' in semester:
            threshold = 40  # Lower threshold for early academic years
        else:
            threshold = 50  # Standard threshold for later years
        
        is_high_risk = risk_score >= threshold
        
        # Calculate confidence based on distance from threshold
        distance = abs(risk_score - threshold)
        confidence = min(95, max(60, 60 + (35 * (distance / (100 - threshold)))))
        
        # If we have critical factors but still below threshold, adjust
        if not is_high_risk and any('Critical:' in f[0] for f in factors):
            is_high_risk = True
            confidence = max(confidence, 80)
        
        # Sort factors by importance (absolute value of weight)
        factors = sorted([f for f in factors if f[0]],  # Remove empty factors
                        key=lambda x: abs(x[1]), 
                        reverse=True)
        
        # Ensure we have a minimum number of factors
        if not factors:
            factors = [('Insufficient data for detailed analysis', 0.5)]
        
        return is_high_risk, confidence, factors[:5]  # Return top 5 factors
        
    except Exception as e:
        logger.error(f"Error in fallback prediction: {e}")
        return True, 75.0, [
            ('System Error', 0.5),
            ('Using Fallback Prediction', 0.5)
        ]

def get_feature_explanations(model, features):
    """Generate explanations for model predictions with more detailed insights."""
    # Feature names and their descriptions
    feature_descriptions = {
        'Last Semester Grade': {
            'description': 'Performance in the most recent semester',
            'impact': 'Higher grades reduce dropout risk',
            'weight': 0.25
        },
        'Overall GPA': {
            'description': 'Cumulative Grade Point Average',
            'impact': 'Higher GPA significantly reduces dropout risk',
            'weight': 0.3
        },
        'Has Job': {
            'description': 'Whether the student is employed',
            'impact': 'Employment can increase time pressure and dropout risk',
            'weight': 0.1
        },
        'Participates in Extracurriculars': {
            'description': 'Involvement in extracurricular activities',
            'impact': 'Participation improves engagement and reduces dropout risk',
            'weight': 0.1
        },
        'Study Hours': {
            'description': 'Weekly study hours',
            'impact': 'More study time generally improves performance',
            'weight': 0.15
        },
        'Attendance': {
            'description': 'Class attendance percentage',
            'impact': 'Higher attendance is strongly correlated with success',
            'weight': 0.2
        }
    }
    
    # If we have feature importance from the model, use it
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features_with_importance = list(zip(feature_descriptions.keys(), importances))
    else:
        # Use default weights from feature_descriptions
        features_with_importance = [
            (name, desc['weight']) 
            for name, desc in feature_descriptions.items()
        ]
    
    # Sort by importance
    features_with_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Generate explanations
    explanations = []
    for feature_name, importance in features_with_importance:
        if feature_name in feature_descriptions:
            desc = feature_descriptions[feature_name]
            explanations.append((
                f"{feature_name}: {desc['impact']}",
                importance
            ))
    
    return explanations

def generate_what_if_scenarios(student_data):
    """Generate what-if scenarios for a student with comprehensive suggestions."""
    scenarios = []
    
    # Scenario 1: Increase attendance
    try:
        # Handle different attendance formats (e.g., '80%', 'Below 40', '40-60%')
        attendance_str = student_data['Attendance'].replace('%', '').split('-')[0].strip()
        if attendance_str.isdigit():
            current_attendance = int(attendance_str)
        elif attendance_str.lower() == 'below 40':
            current_attendance = 30  # Use 30% as a default for 'Below 40'
        else:
            current_attendance = 50  # Default value if format is unexpected
            
        if current_attendance < 80:
            new_data = student_data.copy()
            new_attendance = min(100, current_attendance + 20)  # Increase by 20% or to 100%
            new_data['Attendance'] = f"{new_attendance}%"
            _, new_prob, _ = predict_dropout_risk(new_data)
            current_attendance_display = f"{current_attendance}%" if current_attendance != 30 else "Below 40%"
            scenarios.append({
                'title': '📈 Increase Attendance',
                'description': f'Attend more classes (from {current_attendance_display} to {new_attendance}%)',
                'impact': f'Could reduce dropout risk by {max(5, min(30, int((100 - new_attendance)/3)))}%',
                'action': 'Set reminders for classes and prioritize attendance.'
            })
    except (ValueError, KeyError, AttributeError) as e:
        logger.warning(f"Error processing attendance data: {e}")
        # Skip attendance scenario if there's an error
    
    # Scenario 2: Improve study time
    study_hours_map = {
        '0-1 Hour': '2-3 Hours',
        '2-3 Hours': 'More than 3 Hours',
        'More than 3 Hours': 'More than 3 Hours (with breaks)'
    }
    if student_data['Preparation'] in study_hours_map:
        new_data = student_data.copy()
        new_prep = study_hours_map[student_data['Preparation']]
        new_data['Preparation'] = new_prep
        _, new_prob, _ = predict_dropout_risk(new_data)
        scenarios.append({
            'title': '📚 Optimize Study Time',
            'description': f'Increase study time to {new_prep}',
            'impact': 'Improves understanding and retention of material',
            'action': 'Create a study schedule and stick to it.'
        })
    
    # Scenario 3: Improve grades in specific subjects
    if float(student_data['Overall']) < 3.0:
        scenarios.append({
            'title': '🎯 Target Weak Subjects',
            'description': 'Focus on improving grades in your weakest subjects',
            'impact': 'Significantly improves overall academic performance',
            'action': 'Seek help from professors or tutors in challenging subjects.'
        })
    
    # Scenario 4: Join study groups
    scenarios.append({
        'title': '👥 Join Study Groups',
        'description': 'Participate in peer study sessions',
        'impact': 'Enhances learning through collaboration',
        'action': 'Check department notice boards for study groups.'
    })
    
    # Scenario 5: Time management
    scenarios.append({
        'title': '⏰ Improve Time Management',
        'description': 'Use a planner to organize study and personal time',
        'impact': 'Reduces stress and improves productivity',
        'action': 'Try time-blocking techniques for better schedule management.'
    })
    
    # Ensure we always have at least 3 scenarios
    default_scenarios = [
        {
            'title': '🏋️‍♂️ Maintain Physical Health',
            'description': 'Regular exercise and proper sleep',
            'impact': 'Improves concentration and reduces stress',
            'action': 'Aim for 7-8 hours of sleep and 30 mins of exercise daily.'
        },
        {
            'title': '🍎 Balanced Nutrition',
            'description': 'Eat regular, healthy meals',
            'impact': 'Boosts energy and cognitive function',
            'action': 'Include brain foods like nuts, fish, and fruits in your diet.'
        },
        {
            'title': '📝 Regular Self-Assessment',
            'description': 'Weekly review of progress and challenges',
            'impact': 'Helps identify and address issues early',
            'action': 'Set aside time each week to evaluate your academic progress.'
        }
    ]
    
    # Add default scenarios if we don't have enough
    while len(scenarios) < 3 and default_scenarios:
        scenarios.append(default_scenarios.pop(0))
    
    return scenarios[:6]  # Return maximum of 6 scenarios

def plot_attendance_distribution(df):
    """Generate a plot of attendance distribution."""
    attendance = df['Attendance'].value_counts().sort_index()
    fig = px.bar(
        x=attendance.index,
        y=attendance.values,
        labels={'x': 'Attendance Range', 'y': 'Number of Students'},
        title='Student Attendance Distribution'
    )
    return fig.to_html(full_html=False)

def plot_risk_by_department(df):
    """Generate a plot of dropout risk by department."""
    # Calculate average dropout risk by department
    risk_by_dept = df.groupby('Department')['dropout_risk'].mean().sort_values(ascending=False)
    
    # Create the bar chart
    fig = px.bar(
        x=risk_by_dept.index,
        y=risk_by_dept.values * 100,
        labels={'x': 'Department', 'y': 'Dropout Risk (%)'},
        title='Dropout Risk by Department'
    )
    return fig.to_html(full_html=False)

def plot_performance_vs_risk(df):
    """Generate a scatter plot of performance vs. dropout risk."""
    # Ensure we have the dropout_risk column
    if 'dropout_risk' not in df.columns:
        df['dropout_risk'] = df['Overall'].apply(lambda x: 1 if x < 2.0 else 0)
    
    # Create the scatter plot
    fig = px.scatter(
        df,
        x='Overall',
        y=df['Attendance'].str.extract(r'(\d+)')[0].astype(float),
        color=df['dropout_risk'].astype(str),
        labels={'x': 'Overall GPA', 'y': 'Attendance (%)', 'color': 'Dropout Risk'},
        title='Performance vs. Attendance',
        color_discrete_map={'0': 'green', '1': 'red'}
    )
    return fig.to_html(full_html=False)

# Test route for static files
@app.route('/test/static/<path:filename>')
def test_static(filename):
    return app.send_static_file(filename)

# Chatbot API Endpoint
@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        chatbot = ChatbotService()
        response = chatbot.generate_response(
            user_message=user_message,
            username=current_user.username,
            is_teacher=current_user.is_teacher
        )
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'An error occurred while processing your message',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the app
    app.run(debug=True, port=5000)
