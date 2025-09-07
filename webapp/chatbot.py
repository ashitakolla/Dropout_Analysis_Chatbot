import os
import pandas as pd
from openai import OpenAI
from flask import current_app
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Perplexity Pro client with error handling
try:
    api_key = os.environ.get('PERPLEXITY_API_KEY')
    if not api_key:
        current_app.logger.error("PERPLEXITY_API_KEY not found in environment variables")
        client = None
    else:
        # Configure for Perplexity Pro
        client = OpenAI(
            base_url="https://api.perplexity.ai",
            api_key=api_key,
        )
        # Add headers for Perplexity Pro
        client.headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }
except Exception as e:
    current_app.logger.error(f"Error initializing OpenRouter client: {str(e)}")
    client = None

class ChatbotService:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_path = self.base_dir / 'students_enhanced.csv'
        self.student_data = self._load_student_data()
    
    def _load_student_data(self):
        """Load student data from CSV file."""
        try:
            return pd.read_csv(self.data_path)
        except Exception as e:
            current_app.logger.error(f"Error loading student data: {e}")
            return pd.DataFrame()
    
    def get_student_context(self, username):
        """Get context for a student user."""
        try:
            if self.student_data.empty:
                current_app.logger.warning("Student data is empty")
                return "Student data not available."
            
            student_row = self.student_data[self.student_data['username'] == username]
            if student_row.empty:
                current_app.logger.warning(f"No student found with username: {username}")
                return f"No data available for student: {username}"
                
            student = student_row.iloc[0]
            current_app.logger.info(f"Loaded student data for: {username}")
            
            # Safely get values with defaults
            def safe_get(data, key, default='N/A'):
                try:
                    value = data.get(key, default)
                    return value if pd.notna(value) else default
                except Exception as e:
                    current_app.logger.warning(f"Error getting {key}: {str(e)}")
                    return default
            
            # Create a detailed context string
            context = f"""
            Student Profile:
            - Name: {safe_get(student, 'first_name')} {safe_get(student, 'last_name')}
            - Department: {safe_get(student, 'Department')}
            - Semester: {safe_get(student, 'Semester')}
            - Gender: {safe_get(student, 'Gender')}
            - GPA: {safe_get(student, 'GPA')}
            - Attendance: {safe_get(student, 'Attendance')}%
            - Previous Semester Marks: {safe_get(student, 'PrevSemMarks')}
            - Extracurricular Activities: {safe_get(student, 'Extracurricular')}
            - Study Hours per Day: {safe_get(student, 'StudyHours')}
            - Sleep Hours per Day: {safe_get(student, 'SleepHours')}
            - Social Media Usage: {safe_get(student, 'SocialMedia')} hours/day
            - Internet Access: {safe_get(student, 'Internet')}
            - Family Income: {safe_get(student, 'FamilyIncome')}
            - Parental Education: {safe_get(student, 'ParentEducation')}
            - Dropout Risk: {'High' if float(safe_get(student, 'DropoutRisk', 0)) > 0.5 else 'Low'}
            """
            
            return context
            
        except Exception as e:
            current_app.logger.error(f"Error in get_student_context: {str(e)}", exc_info=True)
            return f"Error loading student data: {str(e)}"
    
    def _safe_stat(self, column, stat_func, default=0):
        """Helper method to safely calculate statistics with error handling."""
        try:
            if column not in self.student_data.columns:
                current_app.logger.warning(f"Column '{column}' not found in student data")
                return default
            return stat_func(self.student_data[column])
        except Exception as e:
            current_app.logger.error(f"Error calculating {column} stat: {str(e)}")
            return default

    def _format_stat(self, value, format_str='.2f'):
        """Helper method to format numeric values."""
        if isinstance(value, (int, float)):
            return f"{value:{format_str}}"
        return 'N/A'

    def get_teacher_context(self):
        """Get context for a teacher user with comprehensive error handling."""
        try:
            if self.student_data.empty:
                current_app.logger.warning("No student data available for teacher context")
                return "Student data not available."
            
            # Calculate basic statistics
            total_students = len(self.student_data)
            
            # Calculate average of 'Last' column as a proxy for GPA
            avg_last_sem = self._safe_stat('Last', lambda x: x.mean(), 0)
            
            # Handle attendance percentage extraction
            def extract_attendance(series):
                try:
                    # Extract the first number from the attendance range (e.g., '80%-100%' -> 80)
                    return series.str.extract(r'(\d+)')[0].astype(float).mean()
                except Exception as e:
                    current_app.logger.error(f"Error extracting attendance: {str(e)}")
                    return 0
                    
            avg_attendance = self._safe_stat('Attendance', extract_attendance, 0)
            
            # Calculate at-risk students based on attendance and grades
            try:
                # Consider students with attendance < 60% or last semester grade < 2.0 as at-risk
                at_risk_count = 0
                if 'Attendance' in self.student_data.columns and 'Last' in self.student_data.columns:
                    attendance_values = self.student_data['Attendance'].str.extract(r'(\d+)')[0].astype(float)
                    at_risk_count = ((attendance_values < 60) | (self.student_data['Last'] < 2.0)).sum()
                    high_risk = (at_risk_count / total_students) * 100
                else:
                    high_risk = 0
                    current_app.logger.warning("Required columns not found for risk calculation")
            except Exception as e:
                current_app.logger.error(f"Error calculating at-risk percentage: {str(e)}")
                high_risk = 0
            
            # Get department statistics safely
            dept_stats_str = "No department data available"
            try:
                if 'Department' in self.student_data.columns:
                    dept_stats = self.student_data['Department'].value_counts().head(3)
                    if not dept_stats.empty:
                        dept_stats_str = '\n'.join([f"- {dept}: {count} students" for dept, count in dept_stats.items()])
            except Exception as e:
                current_app.logger.error(f"Error processing department stats: {str(e)}")
            
            # Format the context string with available statistics
            context = f"""
            Class Statistics:
            - Total Students: {total_students}
            - Average Last Semester Grade: {self._format_stat(avg_last_sem, '.2f')}
            - Average Attendance: {self._format_stat(avg_attendance, '.1f')}%
            - At-Risk Students: {self._format_stat(high_risk, '.1f')}% (attendance < 60% or grade < 2.0)
            
            Top Departments by Enrollment:
            {dept_stats_str}
            """
            
            return context
            
        except Exception as e:
            error_msg = f"Error in get_teacher_context: {str(e)}"
            current_app.logger.error(error_msg, exc_info=True)
            return "Error loading class statistics. Please try again later."
        
        return context
    
    def generate_response(self, user_message, username, is_teacher=False):
        """Generate a response using the OpenAI API."""
        try:
            # Handle simple greetings and common phrases directly
            user_message_lower = user_message.lower().strip()
            simple_greetings = ['hi', 'hello', 'hey', 'hi there', 'hey there', 'greetings']
            
            if user_message_lower in simple_greetings:
                if is_teacher:
                    return "Hello Professor! How can I assist you with your class today?"
                else:
                    return "Hi there! I'm your academic assistant. How can I help you with your studies today?"
            
            # Set up the system prompt based on user role
            if is_teacher:
                current_app.logger.info(f"Generating response for teacher user: {username}")
                context = self.get_teacher_context()
                system_prompt = f"""
                You are an AI teaching assistant for a university. You have access to class statistics 
                and student performance data. Provide helpful, data-driven insights and recommendations 
                to help improve teaching effectiveness and student outcomes. Be professional yet approachable.
                
                Current class statistics:
                {context}
                """
            else:
                current_app.logger.info(f"Generating response for student user: {username}")
                context = self.get_student_context(username)
                system_prompt = f"""
                You are an AI academic advisor for a university student. The student has the following profile:
                {context}
                
                Provide personalized, empathetic, and constructive advice based on the student's profile 
                and their questions. Focus on academic guidance, study tips, and overall well-being.
                Keep your responses clear, concise, and supportive.
                """
            
            current_app.logger.info("Calling Perplexity Pro API...")
            
            # Call the Perplexity Pro API with error handling
            try:
                response = client.chat.completions.create(
                    model="sonar-pro",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=500,  # Reduced for more focused responses
                    top_p=0.9,
                    presence_penalty=0.2  # Using only presence_penalty for better response quality
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Clean up any citation markers from the response
                response_text = response_text.split('[')[0].strip()
                
                current_app.logger.info("Successfully received response from Perplexity Pro")
                return response_text
                
            except Exception as api_error:
                current_app.logger.error(f"Perplexity Pro API error: {str(api_error)}", exc_info=True)
                return "I'm having trouble connecting to the AI service. Please try again in a moment."
            
        except Exception as e:
            error_msg = f"Error generating chatbot response: {str(e)}"
            current_app.logger.error(error_msg, exc_info=True)
            return "I'm sorry, I encountered an error while processing your request. Please try again later."
