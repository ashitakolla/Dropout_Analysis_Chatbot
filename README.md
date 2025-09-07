# Student Dropout Analysis System with AI Assistant

A comprehensive web application that helps predict and prevent student dropouts using machine learning, data analytics, and AI-powered assistance.

## 🌟 Key Features

### Core Functionality
- **Student Dashboard**: Personalized insights and recommendations
- **Teacher Dashboard**: Overview of all students with risk assessment
- **Risk Prediction**: Machine learning model to identify at-risk students
- **What-If Analysis**: Simulate scenarios to improve student outcomes
- **Interactive Visualizations**: Charts and graphs for data exploration

### AI Academic Assistant Chatbot
- **Role-based Assistance**: Different behavior for students and teachers
- **Floating Chat Interface**: Accessible from any page
- **Real-time Responses**: Powered by AI language models
- **Context-Aware**: Uses student/class data to provide relevant responses
- **Responsive Design**: Works on both desktop and mobile devices

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)
- OpenAI API key (for the AI chatbot)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone [your-repository-url]
   cd Dropout_Analysis_Chatbot
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```
   - Configure other environment variables as needed

5. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

## 🤖 AI Chatbot Features

### For Students
- Personalized academic advice and study tips
- Guidance based on individual performance and risk level
- Quick access to important academic information

### For Teachers
- Insights into class-wide statistics
- Student performance and risk factor analysis
- Suggested interventions and teaching strategies

## 🛠 Technical Details

### File Structure
- `app.py`: Main application file
- `chatbot.py`: Core chatbot functionality and API integration
- `static/`: Contains CSS, JavaScript, and other static files
  - `js/chatbot.js`: Frontend chat interface logic
  - `css/chatbot.css`: Styling for the chat interface
- `templates/`: HTML templates
  - `base.html`: Base template with chat interface
  - Other template files...
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not committed to version control)

### Security Notes
- Sensitive data is stored in `.env` and loaded as environment variables
- API keys are never exposed to the frontend
- All external API calls are made server-side

## 📚 Customization

You can customize the chatbot's behavior by modifying the system prompts in `chatbot.py`:
- `get_student_context()`: Controls student data provided to the chatbot
- `get_teacher_context()`: Controls class statistics provided to the chatbot
- `generate_response()`: Contains the system prompts that define the chatbot's behavior

## ❓ Troubleshooting

- **Chatbot not responding**: Check browser's developer console for errors
- **API key issues**: Verify your OpenAI API key in the `.env` file
- **Installation problems**: Ensure all dependencies are installed from `requirements.txt`
- **Connection issues**: Make sure you have an active internet connection for API calls

## 📄 License
[Your license information here]

## 👥 Contributing
Contributions are welcome! Please read our contributing guidelines before submitting pull requests.
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
Dropout_Analysis_Chatbot/
├── webapp/              # Web application code
│   ├── static/          # Static files (CSS, JS, images)
│   │   ├── css/
│   │   ├── js/
│   │   └── img/
│   ├── templates/       # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── student_dashboard.html
│   │   └── teacher_dashboard.html
│   ├── app.py           # Main application file
│   └── enhance_students.py  # Script to enhance student data
├── model_comparison/    # Model comparison results
├── output/              # Output files and model artifacts
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
