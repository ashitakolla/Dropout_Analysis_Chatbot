# Student Dropout Analysis System

A web application that helps predict and prevent student dropouts using machine learning and data analytics.

## Features

- **Student Dashboard**: Personalized insights and recommendations
- **Teacher Dashboard**: Overview of all students with risk assessment
- **Risk Prediction**: Machine learning model to identify at-risk students
- **What-If Analysis**: Simulate scenarios to improve student outcomes
- **Interactive Visualizations**: Charts and graphs for data exploration

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd student-dropout-analysis
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with the following content:
   ```
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=your-secret-key
   ```

5. **Run database migrations**
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

6. **Enhance student data**
   ```bash
   python enhance_students.py
   ```
   This will create `students_enhanced.csv` with additional authentication fields.

## Running the Application

1. **Start the development server**
   ```bash
   flask run
   ```

2. **Access the application**
   Open your browser and go to: http://localhost:5000

## Default Login Credentials

### Teacher Account
- **Username**: teacher
- **Password**: teacher123

### Student Accounts
- Username: First letter of first name + last name (e.g., jsmith)
- Password: Same as username (for demo purposes)

## Project Structure

```
webapp/
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── img/
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── student_dashboard.html
│   └── teacher_dashboard.html
├── app.py               # Main application file
├── enhance_students.py  # Script to enhance student data
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Customization

### Adding New Features
1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request

### Modifying the Model
1. Update the machine learning model in `model.py`
2. Retrain the model using your updated dataset
3. Update the prediction logic in the relevant routes

## Troubleshooting

- **Module not found errors**: Make sure all dependencies are installed
- **Database issues**: Try deleting the database file and re-running migrations
- **Port already in use**: Use `flask run --port 5001` to specify a different port

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Bootstrap 5](https://getbootstrap.com/)
- [Font Awesome](https://fontawesome.com/)
- [Flask](https://flask.palletsprojects.com/)
- [Plotly](https://plotly.com/)
