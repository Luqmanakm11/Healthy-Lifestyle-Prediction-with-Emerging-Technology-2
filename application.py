import streamlit as st
import pandas as pd
import joblib
import dill
import base64
import seaborn as sns
from datetime import datetime
import os
from filelock import FileLock
from openai import OpenAI
import openai
import plotly.express as px
from PIL import Image

# Load API key from Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=api_key)

# File paths
data_file = "Healthy_lifestyle_score_with_healthy_status.csv"
preprocess_path = "data_preprocessor(Classification).pkl"
model_path = "healthy_lifestyle_model(Classification).pkl"

# Read CSV file with validation
def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['date'], on_bad_lines='skip')

        # Ensure consistent column structure
        required_columns = ['user_id', 'date', 'steps', 'active_minutes', 'sleep_hours',
                            'heart_rate_avg', 'calories_burned', 'distance_km', 'healthy_score', 'healthy_status']
        for col in required_columns:
            if col not in data.columns:
                data[col] = 0  # Add missing columns with default values

        return data.dropna(subset=required_columns)
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error



# Write data to CSV file with locking and structure validation
def append_to_csv_file(file_path, data):
    lock_path = f"{file_path}.lock"
    with FileLock(lock_path):
        try:
            # Check if file exists
            if os.path.exists(file_path):
                # Read the header of the existing file
                existing_columns = pd.read_csv(file_path, nrows=0).columns.tolist()

                # Ensure input data matches the existing file's structure
                if not set(existing_columns).issubset(data.columns):
                    raise ValueError("Input data structure does not match the existing file.")

                # Align input data to the existing structure
                data = data.reindex(columns=existing_columns)
            else:
                # If file doesn't exist, initialize with the data's structure
                existing_columns = data.columns.tolist()

            # Append data to the file
            data.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        except Exception as e:
            st.error(f"Error writing to CSV file: {str(e)}")


# Load the preprocessing function (cached)
@st.cache_resource
def load_preprocessor(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessing file not found: {path}")
    with open(path, "rb") as f:
        return dill.load(f)

# Load the trained model (cached)
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

# Ensure model and preprocessor are loaded into session state
def initialize_resources():
    try:
        st.session_state['preprocessor'] = load_preprocessor(preprocess_path)
        st.session_state['model'] = load_model(model_path)
        st.session_state['model_loaded'] = True
    except Exception as e:
        st.error(f"Error loading preprocessing or model: {str(e)}")
        st.session_state['preprocessor'], st.session_state['model'] = None, None
        st.session_state['model_loaded'] = False

# Initialize resources on app start
if 'model_loaded' not in st.session_state:
    initialize_resources()


# CSS for styling
# Add custom CSS for the sidebar menu
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Sidebar Header */
        .sidebar-header {
            font-size: 1.5em;
            font-weight: bold;
            padding: 10px 0;
            color: #4A4A4A;
            text-align: center;
        }

        /* Menu Button Styles */
        .stButton > button {
            width: 100%;
            padding: 15px;
            font-size: 1.2em;
            font-weight: bold;
            color: #4A4A4A;
            background-color: #FFFFFF;
            border: 1px solid #E3E6EB;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        /* Active Button */
        .stButton > button.active {
            background-color: #5C36C9;
            color: white;
        }

        /* Hover Effect */
        .stButton > button:hover {
            background-color: #5C36C9;
            color: white;
        }

        /* Sidebar Footer */
        .sidebar-footer {
            text-align: center;
            margin-top: 20px;
            font-size: 0.9em;
            color: #666666;
        }

    /* Metric Card Styling */
    .metric-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background-color: white;
        border: 1px solid #E3E6EB;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .metric-title {
        font-weight: bold;
        font-size: 14px;
        color: #6B7280;
        margin-bottom: 5px;
    }
    .metric-value {
        font-weight: bold;
        font-size: 24px;
        color: #4A4A4A;
    }
    </style>
    """, unsafe_allow_html=True)



# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None


# Login Page
def login_page():
    st.title("💜 FitnessDashboard")
    st.subheader("Please enter your User ID to continue")
    user_id = st.text_input("User ID", placeholder="e.g., 12345")

    if st.button("Submit"):
        if user_id.strip():
            # Save user_id in session state
            st.session_state['user_id'] = user_id.strip()
            st.success("User ID saved. Redirecting...")

            # Redirect to the dashboard
            st.rerun()  # Reloads the app to show the dashboard
        else:
            st.error("User ID cannot be empty.")



# Sidebar Menu Function
def sidebar_menu():
    logo_path = "BeHealthy_Logo(2).png"  # Replace with your logo path

    # Check if the image exists
    if os.path.exists(logo_path):
        try:
            # Read the image file and encode it to base64
            with open(logo_path, "rb") as img_file:
                img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()

            # Display the logo and center it using HTML
            st.sidebar.markdown(f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{img_base64}" width="150">
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Error displaying image: {str(e)}")
    else:
        st.sidebar.error("Logo image not found.")

    # Sidebar Header
    st.sidebar.markdown('<div class="sidebar-header"></div>', unsafe_allow_html=True)

    # Initialize Active Page in Session State
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Home"  # Default page is Home

    # Sidebar Buttons
    if st.sidebar.button("Home", key="home_btn"):
        st.session_state["active_page"] = "Home"

    if st.sidebar.button("Dashboard", key="dashboard_btn"):
        st.session_state["active_page"] = "Dashboard"

    if st.sidebar.button("Predict", key="predict_btn"):
        st.session_state["active_page"] = "Predict"

    if st.sidebar.button("About", key="about_btn"):
        st.session_state["active_page"] = "About"

    # Sidebar Footer
    st.sidebar.markdown('<div class="sidebar-footer">&copy; 2025 My Company</div>', unsafe_allow_html=True)

    return st.session_state["active_page"]


def home_page():
    st.title("🏡 Welcome to BeHealthy")
    st.subheader("Stay Updated on the Latest Health News!")

    # Add some introductory text
    st.markdown("""
    Welcome to BeHealthy! Here, you will find the latest articles, tips, and insights on living a healthier life. 
    Explore articles about fitness, nutrition, mental well-being, and much more. Stay informed and motivated to achieve your health goals!
    """)

    # News Section (you can replace the URLs and titles with actual data)
    st.markdown("### Latest Healthy Lifestyle News")

    # News 1
    st.markdown("#### 📰 **How to Stay Active During Winter**")
    st.image("https://www.cdc.gov/physical-activity/media/images/Personinwheelchairplayinginsnowistock16x9.jpg", caption="Stay Active During Winter", use_column_width=True)
    st.markdown("""
    The winter season can be a challenging time to stay active, with colder temperatures, slippery conditions, and fewer daylight hours. 
    But staying physically active is one of the best ways to improve your mental and physical health and keep on track with your fitness goals.
    [Read more →](https://www.cdc.gov/physical-activity/features/stay-active-this-winter.html)
    """)

    # News 2
    st.markdown("#### 📰 **5 of the best exercises you can ever do**")
    st.image("https://domf5oio6qrcr.cloudfront.net/medialibrary/7194/3448d6cd-d40d-456c-b96d-0ef201c7dcac.jpg", caption="Workout for healthier lifestyle", use_column_width=True)
    st.markdown("""
    If you're not an athlete or serious exerciser — and you just want to work out for your health or to fit in your clothes better — the gym scene can be intimidating and overwhelming. What are the best exercises for me? How will I find the time?
    [Read more →](https://www.health.harvard.edu/staying-healthy/5-of-the-best-exercises-you-can-ever-do)
    """)

    # News 3
    st.markdown("#### 📰 **How to control your stress**")
    st.image("https://media.licdn.com/dms/image/C4D12AQGbyPexOOuzkA/article-cover_image-shrink_720_1280/0/1598376885705?e=2147483647&v=beta&t=lrtntu4J44wU6mpo_xYvCwtu6VZCCmM4ogRtVCDnWLo", caption="Stress Management", use_column_width=True)
    st.markdown("""
    Many people deal with stress every day. Work, family issues, health concerns, and financial obligations are parts of everyday life that commonly contribute to heightened stress levels.
    [Read more →](https://www.healthline.com/nutrition/16-ways-relieve-stress-anxiety)
    """)


def dashboard_page():
    st.title("📊 Dashboard")
    st.subheader(f"Welcome, User {st.session_state['user_id']}!")

    # Refresh button
    if st.button('Refresh Data'):
        st.experimental_rerun()  # This will refresh the page and load the latest data

    if not os.path.exists(data_file):
        st.error("No data file found. Please enter some predictions first.")
        return

    # Load user data
    data = pd.read_csv(data_file)

    # Ensure 'date' column is datetime-like
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date'])  # Drop rows with invalid dates

    # Convert user_id to string for consistency
    data['user_id'] = data['user_id'].astype(str)

    # Ensure session user_id is in the same format (convert to string)
    user_id = str(st.session_state['user_id'])

    # Filter the user data based on the user_id
    user_data = data[data['user_id'] == user_id]

    if user_data.empty:
        st.warning("No historical data found for your User ID.")
        return

    # Sidebar Filter Pane
    with st.sidebar:
        st.header("Filter Pane")

        # Ensure date-related filters only use valid data
        available_years = user_data['date'].dt.year.unique()
        year = st.selectbox("Select Year", available_years)

        available_months = user_data[user_data['date'].dt.year == year]['date'].dt.month_name().unique()
        month = st.selectbox("Select Month", available_months)

        user_data = user_data[
            (user_data['date'].dt.year == year) &
            (user_data['date'].dt.month_name() == month)
            ]

    if user_data.empty:
        st.warning("No data available for the selected year and month.")
        return

    # Metrics Summary
    st.markdown("### Average Values")
    avg_metrics = {
        "Steps": user_data['steps'].mean(),
        "Active Minutes": user_data['active_minutes'].mean(),
        "Sleep Hours": user_data['sleep_hours'].mean(),
        "Heart Rate": user_data['heart_rate_avg'].mean(),
        "Calories": user_data['calories_burned'].mean(),
        "Distance (km)": user_data['distance_km'].mean(),
    }

    # Display metrics in rows of 3
    metrics = list(avg_metrics.items())
    for i in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col, (metric, value) in zip(cols, metrics[i:i + 3]):
            with col:
                st.markdown(f"<div class='metric-box'>"
                            f"<div class='metric-title'>{metric}</div>"
                            f"<div class='metric-value'>{value:.2f}</div>"
                            "</div>", unsafe_allow_html=True)

    # Graphs
    st.markdown("### Performance Metrics")
    graphs = [
        {"title": "Step Performance", "x": "date", "y": "steps", "color": "#6F2DBD", "plot_type": "bar"},
        {"title": "Active Minutes", "x": "date", "y": "active_minutes", "color": "#9D4EDD", "plot_type": "line"},
        {"title": "Heart Rate Average", "x": "date", "y": "heart_rate_avg", "color": "#A26DD5", "plot_type": "line"},
        {"title": "Calories Burned", "x": "date", "y": "calories_burned", "color": "#F72585", "plot_type": "bar"},
        {"title": "Distance in Kilometers", "x": "date", "y": "distance_km", "color": "#4895EF", "plot_type": "line"},
        {"title": "Sleep Hours", "x": "date", "y": "sleep_hours", "color": "#7209B7", "plot_type": "bar"},
    ]

    # Arrange graphs in 1 column per row
    for graph in graphs:
        st.markdown(f"#### {graph['title']}")  # Display custom title

        if graph["plot_type"] == "bar":
            fig = px.bar(user_data, x=graph["x"], y=graph["y"], color_discrete_sequence=[graph["color"]])
        elif graph["plot_type"] == "line":
            fig = px.line(user_data, x=graph["x"], y=graph["y"], line_shape="linear", markers=True,
                          color_discrete_sequence=[graph["color"]])

        # Hide title in the graph
        fig.update_layout(title='')  # Set title to empty string to hide it

        # Show interactive plot with hover info
        st.plotly_chart(fig)


# Define a function to get suggestions from ChatGPT
def get_suggestions(input_data, health_status):
    try:
        # Safely access data fields
        date = input_data.get('date', 'N/A')
        mood = input_data.get('mood', 'N/A')
        workout_type = input_data.get('workout_type', 'N/A')
        weather_conditions = input_data.get('weather_conditions', 'N/A')
        location = input_data.get('location', 'N/A')
        steps = input_data.get('steps', 0)
        calories_burned = input_data.get('calories_burned', 0)
        distance_km = input_data.get('distance_km', 0)
        active_minutes = input_data.get('active_minutes', 0)
        sleep_hours = input_data.get('sleep_hours', 0)
        heart_rate_avg = input_data.get('heart_rate_avg', 0)

        # Create a structured message for ChatGPT
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in providing personalized lifestyle improvement advice."},
            {"role": "user", "content": f"""
                Based on the following details:
                - Date: {date}
                - Mood: {mood}
                - Workout Type: {workout_type}
                - Weather Conditions: {weather_conditions}
                - Location: {location}
                - Steps: {steps}
                - Calories Burned: {calories_burned}
                - Distance (km): {distance_km}
                - Active Minutes: {active_minutes}
                - Sleep Hours: {sleep_hours}
                - Heart Rate (Average): {heart_rate_avg}

                The current healthy lifestyle is considered {health_status} based on above input.
                Provide actionable suggestions to improve the lifestyle to become healthier.
            """}
        ]

        # Make a call to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        # Extract and return the response text
        response_message = response.choices[0].message.content.strip()
        return response_message

    except Exception as e:
        return f"Error generating suggestions: {str(e)}"


# Prediction page
def predict_page():
    st.title("\U0001F52E Predict Your Healthy Lifestyle")
    st.write("Enter your details on the left to predict your healthy lifestyle score. Results will appear on the right.")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Input Details")
        # Input fields
        date = st.date_input("Date")
        mood = st.selectbox("Mood", ["Stressed", "Tired", "Neutral", "Happy"])
        workout_type = st.selectbox("Workout Type", ["No Workout", "Walking", "Cycling", "Yoga", "Gym Workout", "Running", "Swimming"])
        weather_conditions = st.selectbox("Weather Conditions", ["Fog", "Rain", "Snow", "Clear"])
        location = st.selectbox("Location", ["Other", "Home", "Office", "Park", "Gym"])
        steps = st.number_input("Steps", min_value=0.0, step=1.0)
        calories_burned = st.number_input("Calories Burned", min_value=0.0, step=0.1)
        distance_km = st.number_input("Distance (km)", min_value=0.0, step=0.1)
        active_minutes = st.number_input("Active Minutes", min_value=0.0, step=1.0)
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.1)
        heart_rate_avg = st.number_input("Heart Rate (Average)", min_value=0.0, step=1.0)

    with col2:
        st.markdown("### Predicted Results")
        prediction_placeholder = st.empty()
        health_status_placeholder = st.empty()
        suggestion_placeholder = st.empty()

    if st.button("Predict"):
        if not st.session_state.get('model_loaded', False):
            st.error("Model or preprocessor not loaded.")
            return

        try:
            # Prepare the data
            input_data = pd.DataFrame([{
                'user_id': st.session_state.get('user_id', 'unknown_user'),
                'date': date.strftime('%m/%d/%Y'),
                'mood': mood,
                'workout_type': workout_type,
                'weather_conditions': weather_conditions,
                'location': location,
                'steps': steps,
                'calories_burned': calories_burned,
                'distance_km': distance_km,
                'active_minutes': active_minutes,
                'sleep_hours': sleep_hours,
                'heart_rate_avg': heart_rate_avg,
            }])

            input_data_before_preprocess=input_data.copy()

            # Preprocess input data
            preprocessor = st.session_state['preprocessor']
            preprocessed_data = preprocessor.preprocess(input_data)

            # Predict
            model = st.session_state['model']
            prediction = model.predict(preprocessed_data)
            healthy_status = prediction[0]
            health_status = 'Healthy' if healthy_status == 1 else 'Unhealthy'

            # Display results
            # prediction_placeholder.success(f"Healthy Score: {healthy_status}")
            health_status_placeholder.success(f"Health Status: {health_status}")

            # Get improvement suggestions from ChatGPT
            suggestions = get_suggestions(input_data_before_preprocess, health_status)
            suggestion_placeholder.info(f"Suggestions: {suggestions}")

            # Save to file
            input_data_before_preprocess['healthy_status'] = health_status

            # Align input_data with the existing file structure
            if os.path.exists(data_file):
                # Read existing file columns
                existing_columns = pd.read_csv(data_file, nrows=0).columns.tolist()
                # Align input_data to match existing structure
                input_data = input_data.reindex(columns=existing_columns, fill_value=0)

            # Save data with structure alignment
            append_to_csv_file(data_file, input_data_before_preprocess)
            st.success("Prediction saved successfully!")


        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")


# About Page Function
def about_page():
    st.title("👩‍💻 About This App")

    # Section 1: Developer's Info
    st.markdown("### Developer")

    # Display Developer's Image
    try:
        dev_image = Image.open("Luqman_Image.jpg")  # Replace with your image path
        st.image(dev_image, width=150, caption="Luqman", use_column_width=False)
    except Exception as e:
        st.warning("Developer image not found. Please add 'developer_image.jpg' to the app folder.")

    # Developer Info
    st.markdown("""
    - **Name**: Luqmanul Hakim Bin Abdul Latif  
    - **Matric Number**: U2100482
    - **Email**: luqman101103@gmail.com  
    """)

    # Section 2: App Aim and Objective
    st.markdown("### Aim and Objective")
    st.info("""
    This app is designed to help users **track their daily lifestyle habits** and **make informed decisions to improve their overall well-being**. 

    With the rise of fitness trackers and smart devices, people now have access to a wealth of personal health data. However, without meaningful insights, it can be difficult to interpret this data and take actionable steps to live healthier lives.

    The core objectives of this app are:
    - To **analyze users' daily fitness data** (e.g., steps, calories burned, active minutes, sleep hours).
    - To **predict users' health status** using a machine learning model.
    - To **provide personalized suggestions** for improving lifestyle habits based on users' activity patterns.
    - To **track progress over time** through an interactive dashboard.

    By combining modern technologies such as **machine learning** and **AI-generated insights**, this app empowers users to **adopt healthier routines** and **achieve their fitness goals** in a more personalized and effective way.
    """)

    # Section 3: Technologies Used
    st.markdown("### Technologies Used")
    st.markdown("""
    - **Streamlit**: For building the web app interface.
    - **Scikit-learn**: For training the machine learning model.
    - **Joblib & Dill**: For saving and loading the model and preprocessor.
    - **Plotly**: For creating interactive graphs.
    - **OpenAI API**: For generating personalized health improvement suggestions.
    """)

    # Section 4: Acknowledgments
    st.markdown("### Acknowledgments")
    st.markdown("""
    - **Supervisor**: Assoc. Prof. Dr. Suraya Hamid 
    - **Friends and Family**: For their continuous support throughout the project.
    """)

# Main App Flow
add_custom_css()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None  # Example initialization

if st.session_state["user_id"] is None:
    login_page()  # This is where your login page function is called
else:
    # Use the Sidebar Menu
    active_page = sidebar_menu()

    # Render the Selected Page
    if active_page == "Home":
        home_page()  # Home Page
    elif active_page == "Dashboard":
        dashboard_page()  # Dashboard Page
    elif active_page == "Predict":
        predict_page()  # Predict Page
    elif active_page == "About":
        about_page()