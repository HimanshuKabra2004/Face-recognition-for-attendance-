import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

nimgs = 10

datetoday = date.today().strftime("%m_%d_%y")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def clear_attendance():
    """Clears the attendance records by resetting the CSV file."""
    filepath = f'Attendance/Attendance-{datetoday}.csv'
    if os.path.exists(filepath):
        with open(filepath, 'w') as f:
            # Write only the header (reset the file)
            f.write('Name,Roll,Time\n')
        return f"Attendance records for {datetoday} cleared."
    else:
        return f"No attendance records found for {datetoday} to clear."

# Example usage
@app.route('/clear_attendance', methods=['POST'])
def clear_attendance_route():
    """Endpoint to clear attendance records."""
    message = clear_attendance()
    names, rolls, times, l = [], [], [], 0  # Reset display data
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), mess=message)


def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(
            gray, 1.2, 5, minSize=(20, 20))
        return face_points
    return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# def extract_attd():
#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     names = df['Name']
#     rolls = df['Roll']
#     times = df['Time']
#     l = len(df)
#     return names, rolls, times, l
def extract_attd():
    # Ensure that the attendance file exists
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        names = df['Name'].tolist()
        rolls = df['Roll'].tolist()
        times = df['Time'].tolist()
        l = len(df)
        return names, rolls, times, l
    except FileNotFoundError:
        return [], [], [], 0



def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    if userid not in df['Roll'].astype(str).values:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_date},{current_time}')



# @app.route('/')
# @app.route('/')
# def home():
#     names, rolls, times, l = extract_attd()
#     attendance_data = zip(names, rolls, times)  
#     return render_template('home.html', attendance_data=attendance_data, l=l, totalreg=totalreg())
@app.route('/')
def home():
    # Ensure attendance file exists for today, otherwise return an empty table
    if not os.path.exists(f'Attendance/Attendance-{datetoday}.csv'):
        names, rolls, times, l = [], [], [], 0
    else:
        # Extract attendance if file exists
        names, rolls, times, l = extract_attd()

    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())




@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), mess='No trained model found. Please add a new face to continue.')
    
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attd()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs * 5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()  
    names, rolls, times, l = extract_attd()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

if __name__ == '__main__':
    app.run(debug=True)
