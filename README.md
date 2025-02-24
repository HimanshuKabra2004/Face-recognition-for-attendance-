# **Face Recognition Attendance System**

## **Overview**
This project is an **automated attendance system** that uses **face recognition** technology to mark attendance in real-time. It leverages **OpenCV, Flask, NumPy, Pandas, Scikit-learn, and Joblib** to detect, recognize, and store attendance data efficiently.

## **Features**
- **Face Detection**: Identifies faces using OpenCV’s Haar Cascade.
- **Face Recognition**: Uses a K-Nearest Neighbors (KNN) model for identifying individuals.
- **Real-time Attendance Marking**: Detects faces and records attendance in a CSV file.
- **User Registration**: Allows new users to register their face.
- **Web Interface**: Flask-based UI for managing attendance records.
- **CSV Storage**: Stores attendance logs with timestamps.

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: OpenCV, Flask, NumPy, Pandas, Scikit-learn, Joblib
- **Machine Learning Model**: KNN Classifier
- **Database**: CSV-based storage

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/face-recognition-attendance.git
cd face-recognition-attendance
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Application**
```bash
python i221.py
```
The application will start and be accessible at `http://127.0.0.1:5000/`.

## **Usage**
1. **Add New Users**: Navigate to `/add` and register a new face.
2. **Start Recognition**: Visit `/start` to begin detecting and marking attendance.
3. **View Attendance Records**: Attendance is logged in `Attendance/Attendance-<date>.csv`.
4. **Clear Attendance**: Visit `/clear_attendance` to reset daily attendance logs.

## **File Structure**
```
face-recognition-attendance/
│── Attendance/              # Stores daily attendance CSV files
│── static/
│   ├── faces/               # Stores registered face images
│   ├── face_recognition_model.pkl  # Trained model
│── templates/
│   ├── home.html            # Web interface template
│── i221.py                  # Main application file
│── requirements.txt         # List of dependencies
│── README.md                # Project documentation
```

## **Future Improvements**
- Enhance accuracy using **Deep Learning (CNNs)**.
- Store attendance data in **a database (MySQL/PostgreSQL)**.
- Implement **liveness detection** to prevent spoofing.
- Develop a **cloud-based** version for remote access.

## **Contributors**
- **Your Name** (Developer)
- Contributions are welcome! Feel free to submit **pull requests**.

## **License**
This project is licensed under the **MIT License**.

---
Let me know if you need any modifications or additions! 🚀
