import pandas as pd
import numpy as np
import hashlib
import secrets
import string
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_password(username, length=12):
    """Generate a consistent password based on username."""
    # Use a fixed secret salt for consistency
    salt = "edumonitor123"
    # Create a hash of the username + salt
    hash_obj = hashlib.sha256((username + salt).encode())
    # Take first 8 characters of the hash and add some special characters
    base = hash_obj.hexdigest()[:8]
    # Ensure we have at least one of each required character type
    return f"{base[0].upper()}{base[1:3]}!{base[3:5]}@1"

def generate_username(first_name, last_name, existing_usernames):
    """Generate a unique username based on first and last name."""
    base = f"{first_name[0].lower()}{last_name.lower()}"
    username = base
    counter = 1
    
    while username in existing_usernames:
        username = f"{base}{counter}"
        counter += 1
    
    return username

def enhance_student_data(input_file, output_file):
    """Enhance student data with usernames and hashed passwords."""
    # Read the original data
    df = pd.read_csv(input_file)
    
    # Generate realistic first and last names based on gender
    first_names_male = ['Rahul', 'Amit', 'Vikram', 'Rajesh', 'Suresh', 'Ankit', 'Rohit', 'Vivek', 'Prakash', 'Dinesh']
    first_names_female = ['Priya', 'Anjali', 'Sneha', 'Divya', 'Pooja', 'Neha', 'Meera', 'Kavita', 'Sunita', 'Rekha']
    last_names = ['Sharma', 'Verma', 'Gupta', 'Patel', 'Yadav', 'Kumar', 'Singh', 'Mishra', 'Jain', 'Choudhary']
    
    # Generate first and last names
    df['first_name'] = df['Gender'].apply(
        lambda x: np.random.choice(first_names_male if x == 'Male' else first_names_female)
    )
    df['last_name'] = np.random.choice(last_names, size=len(df))
    
    # Generate unique usernames
    existing_usernames = set()
    df['username'] = df.apply(
        lambda row: generate_username(row['first_name'], row['last_name'], existing_usernames),
        axis=1
    )
    
    # Generate consistent passwords based on usernames
    df['password'] = df['username'].apply(generate_password)
    
    # Create a secure hash of the password (in a real app, use a proper password hashing library)
    df['password_hash'] = df['password'].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()
    )
    
    # Reorder columns to put ID and authentication info first
    cols = ['username', 'password', 'password_hash', 'first_name', 'last_name'] + \
           [col for col in df.columns if col not in ['username', 'password', 'password_hash', 'first_name', 'last_name']]
    df = df[cols]
    
    # Save the enhanced dataset
    df.to_csv(output_file, index=False)
    
    # Save a separate file with just the credentials (for reference)
    credentials = df[['username', 'password', 'first_name', 'last_name']]
    credentials.to_csv('student_credentials.csv', index=False)
    
    return df

if __name__ == "__main__":
    input_file = 'students.csv'
    output_file = 'students_enhanced.csv'
    
    print(f"Enhancing student data from {input_file}...")
    enhanced_df = enhance_student_data(input_file, output_file)
    print(f"Enhanced data saved to {output_file}")
    print(f"Student credentials saved to student_credentials.csv")
    print(f"\nSample data:")
    print(enhanced_df[['username', 'first_name', 'last_name', 'Department', 'Gender']].head())
