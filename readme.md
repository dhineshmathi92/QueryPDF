
# Project Title

This project provides a question-and-answer application based on the `main.py` backend and `index.html` frontend. Follow the steps below to set up and run the application.

## Prerequisites

Ensure you have the following installed on your system:

1. Python 3.7 or higher
2. pip (Python package manager)

## Setup Instructions

1. **Navigate to the Project Folder**

   Open the command prompt (cmd) and navigate to the project folder using the `cd` command:

   ```
   cd path/to/project/folder
   ```

2. **Activate the Virtual Environment**

   Activate the virtual environment using the following command:

   ```
   .venv\Scripts\activate
   ```

   *Note:* Ensure the `.venv` folder exists in your project directory. If not, create a virtual environment using:

   ```
   python -m venv .venv
   ```

3. **Install Necessary Libraries**

   Install the required Python libraries by running:

   ```
   pip install -r requirements.txt
   ```

4. **Run the Application**

   Start the application backend by running:

   ```
   uvicorn main:app --reload
   ```

   The server will start, and you will see a URL (e.g., `http://127.0.0.1:8000`) where the backend is running.

5. **Access the Frontend**

   Open a web browser and navigate to the `index.html` file. You can do this by entering the following in the browser's address bar:

   ```
   file:///path/to/project/folder/index.html
   ```

   Replace `path/to/project/folder` with the actual path to the project directory.

6. **Start Asking Questions**

   Once the frontend is loaded, you can start interacting with the application by asking questions.


## Troubleshooting

If you encounter any issues, ensure that:

1. All dependencies are correctly installed.
2. The virtual environment is activated.
3. The backend server is running without errors.
4. The browser correctly accesses the `index.html` file.


