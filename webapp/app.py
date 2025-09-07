import os
import hashlib
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
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
DATA_PATH = Path('students_enhanced.csv')
MODEL_PATH = Path('output/best_model.pkl')

# User class for Flask-Login
class User:
    def __init__(self, username, password_hash, first_name, last_name, is_teacher=False):
        self.username = username
        self.password_hash = password_hash
        self.first_name = first_name
        self.last_name = last_name
        self.is_teacher = is_teacher
        self.is_authenticated = True
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
        creds_df = pd.read_csv('student_credentials.csv')
        
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

# Initialize model as None
model = None

# Get the absolute path to the model file
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')

# Try to load the trained model if the file exists
try:
    if os.path.exists(MODEL_PATH):
        logger.info(f"Attempting to load model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("Successfully loaded the trained model")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Checking alternative locations...")
        # Try alternative model paths
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_comparison', 'best_model.pkl')
        if os.path.exists(alt_path):
            logger.info(f"Found model at alternative location: {alt_path}")
            model = joblib.load(alt_path)
            logger.info("Successfully loaded the trained model from alternative location")
        else:
            logger.warning("No model file found. Running in demo mode with limited functionality.")
except Exception as e:
    logger.error(f"Error loading the model: {str(e)}", exc_info=True)
    model = None

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
    return render_template('index.html')

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
                login_user(user)
                next_page = request.args.get('next')
                logger.info(f"User {username} logged in successfully")
                return redirect(next_page or url_for('index'))
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
    
    # Load student data
    df = pd.read_csv(DATA_PATH)
    
    # Find the student by username
    student_data = df[df['username'] == current_user.username]
    
    # If no student found with that username, try to find by name
    if student_data.empty:
        student_data = df[df['first_name'].str.lower() + df['last_name'].str.lower() == current_user.username.lower()]
    
    if student_data.empty:
        flash('Student data not found', 'danger')
        return redirect(url_for('logout'))
        
    student_data = student_data.iloc[0].to_dict()
    
    # Generate predictions and explanations
    prediction, confidence, explanations = predict_dropout_risk(student_data)
    what_if_scenarios = generate_what_if_scenarios(student_data)
    
    # Get department stats
    department = student_data.get('Department', 'Unknown')
    department_stats = {
        'total_students': len(df[df['Department'] == department]) if 'Department' in df.columns else 0,
        'avg_attendance': df[df['Department'] == department]['Attendance'].str.extract(r'(\d+)')[0].astype(float).mean() if 'Attendance' in df.columns else 0,
        'avg_overall': df[df['Department'] == department]['Overall'].mean() if 'Overall' in df.columns else 0
    }
    
    return render_template(
        'student_dashboard.html',
        student=student_data,
        prediction=prediction,
        confidence=confidence,
        explanations=explanations,
        what_if_scenarios=what_if_scenarios,
        department_stats=department_stats
    )

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
def predict_dropout_risk(student_data):
    """
    Predict dropout risk for a student using the trained model or fallback heuristics.
    Returns: (is_high_risk, confidence, explanations)
    """
    # Try to use the model if available
    if model is not None:
        try:
            features = preprocess_student_data(student_data)
            proba = model.predict_proba([features])[0]
            dropout_prob = proba[1]  # Probability of class 1 (dropout)
            prediction = dropout_prob > 0.5
            confidence = max(proba) * 100
            
            # Generate explanations
            explanations = get_feature_explanations(model, features)
            return prediction, confidence, explanations
            
        except Exception as e:
            logger.error(f"Error making model prediction: {e}")
    
    # Fallback to heuristic-based prediction if model fails or is not available
    try:
        # Calculate risk based on key factors
        risk_score = 0
        factors = []
        
        # Attendance (up to 40 points)
        attendance = student_data.get('Attendance', '0%')
        if 'Below 40' in attendance:
            risk_score += 40
            factors.append(('Attendance', 0.4))
        elif '%' in attendance:
            pct = int(attendance.replace('%', '').split('-')[0])
            if pct < 60:
                risk_score += 30
                factors.append(('Attendance', 0.3))
            elif pct < 80:
                risk_score += 15
                factors.append(('Attendance', 0.15))
        
        # Overall GPA (up to 30 points)
        gpa = float(student_data.get('Overall', 0))
        if gpa < 2.0:
            risk_score += 30
            factors.append(('GPA', 0.3))
        elif gpa < 2.5:
            risk_score += 15
            factors.append(('GPA', 0.15))
        
        # Study time (up to 20 points)
        study_time = student_data.get('Preparation', '')
        if study_time in ['0-1 Hour', '2-3 Hours']:
            risk_score += 20
            factors.append(('Study Time', 0.2))
        
        # Extracurriculars (up to 10 points)
        if student_data.get('Extra', '').lower() == 'no':
            risk_score += 10
            factors.append(('No Extracurriculars', 0.1))
        
        # Determine final prediction (threshold at 40 points)
        is_high_risk = risk_score >= 40
        confidence = min(95, 50 + abs(risk_score - 40))  # Confidence based on distance from threshold
        
        # Sort factors by importance
        factors.sort(key=lambda x: x[1], reverse=True)
        
        return is_high_risk, confidence, factors
        
    except Exception as e:
        logger.error(f"Error in fallback prediction: {e}")
        # Last resort fallback
        return True, 75.0, [
            ('System Error', 0.5),
            ('Using Fallback Prediction', 0.5)
        ]

def get_feature_explanations(model, features):
    """Generate explanations for model predictions."""
    if hasattr(model, 'feature_importances_'):
        feature_names = get_feature_names()
        importances = model.feature_importances_
        explanations = list(zip(feature_names, importances))
        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:5]  # Top 5 most important features
    
    # Default explanations if feature importances aren't available
    return [
        ('Attendance', 0.35),
        ('Overall GPA', 0.25),
        ('Study Time', 0.15),
        ('Extracurricular Activities', 0.1),
        ('Department', 0.08)
    ]

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

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the app
    app.run(debug=True)
