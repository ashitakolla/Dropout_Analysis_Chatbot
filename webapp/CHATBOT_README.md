# AI Academic Assistant Chatbot

This feature adds an AI-powered chatbot to the Student Dropout Analysis system, providing personalized academic assistance to students and data-driven insights to teachers.

## Features

- **Role-based Assistance**: Different behavior for students and teachers
- **Floating Chat Interface**: Accessible from any page
- **Real-time Responses**: Powered by OpenAI's GPT-3.5-turbo
- **Context-Aware**: Uses student/class data to provide relevant responses
- **Responsive Design**: Works on both desktop and mobile devices

## Setup Instructions

1. **Get an OpenAI API Key**
   - Sign up at [OpenAI](https://platform.openai.com/signup) if you don't have an account
   - Create an API key in the [API Keys](https://platform.openai.com/account/api-keys) section

2. **Configure Environment Variables**
   - Edit the `.env` file in the webapp directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

## How It Works

### For Students
- The chatbot has access to the student's academic profile
- Provides personalized academic advice and study tips
- Offers guidance based on the student's performance and risk level

### For Teachers
- The chatbot has access to class-wide statistics
- Provides insights into student performance and risk factors
- Suggests interventions and teaching strategies

## Files Added/Modified

- `chatbot.py`: Core chatbot functionality and API integration
- `static/js/chatbot.js`: Frontend chat interface logic
- `static/css/chatbot.css`: Styling for the chat interface
- `templates/base.html`: Added chat interface HTML
- `.env`: Added OpenAI API key configuration
- `requirements.txt`: Added OpenAI package dependency

## Security Notes

- The OpenAI API key is stored in the `.env` file and loaded as an environment variable
- The key is never exposed to the frontend
- All API calls are made server-side

## Customization

You can customize the chatbot's behavior by modifying the system prompts in `chatbot.py`:
- `get_student_context()`: Controls what student data is provided to the chatbot
- `get_teacher_context()`: Controls what class statistics are provided to the chatbot
- `generate_response()`: Contains the system prompts that define the chatbot's behavior

## Troubleshooting

- If the chatbot isn't responding, check the browser's developer console for errors
- Verify that your OpenAI API key is correctly set in the `.env` file
- Ensure you have an active internet connection to access the OpenAI API
