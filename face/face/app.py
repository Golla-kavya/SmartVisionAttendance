# app.py - Smart Attendance System with Absent Dates Tracking
from flask import Flask, render_template, request, jsonify
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime, timedelta
import json
import base64
from collections import defaultdict

app = Flask(__name__)

# Create necessary directories
os.makedirs('known_faces', exist_ok=True)
os.makedirs('attendance_records', exist_ok=True)

# Global variables
known_face_encodings = []
known_face_names = []

# Available subjects
SUBJECTS = [
    'Mathematics',
    'Physics',
    'Chemistry',
    'English',
    'Computer Science',
    'Biology',
    'History',
    'Geography'
]

# Security Questions
SECURITY_QUESTIONS = [
    "What is your mother's maiden name?",
    "What was the name of your first pet?",
    "What city were you born in?",
    "What is your favorite movie?",
    "What was your childhood nickname?",
    "What is the name of your favorite teacher?",
    "What street did you grow up on?",
    "What is your favorite food?"
]

# Demo user database
STUDENTS = {
    'john_doe': {
        'name': 'John Doe', 
        'usn': 'CS001', 
        'password': 'student123',
        'security_question': "What is your mother's maiden name?",
        'security_answer': 'smith'
    },
    'jane_smith': {
        'name': 'Jane Smith', 
        'usn': 'CS002', 
        'password': 'student123',
        'security_question': "What was the name of your first pet?",
        'security_answer': 'fluffy'
    },
    'bob_johnson': {
        'name': 'Bob Johnson', 
        'usn': 'CS003', 
        'password': 'student123',
        'security_question': "What city were you born in?",
        'security_answer': 'bangalore'
    }
}

TEACHERS = {
    'prof_kumar': {
        'name': 'Prof. Kumar', 
        'password': 'teacher123', 
        'subject': 'Mathematics',
        'security_question': "What is your favorite movie?",
        'security_answer': 'inception'
    },
    'prof_sharma': {
        'name': 'Prof. Sharma', 
        'password': 'teacher123', 
        'subject': 'Physics',
        'security_question': "What was your childhood nickname?",
        'security_answer': 'rocky'
    }
}

# Demo attendance data
STUDENT_ATTENDANCE = {
    'john_doe': {
        'Mathematics': {'present': 28, 'total': 30},
        'Physics': {'present': 25, 'total': 30},
        'Chemistry': {'present': 27, 'total': 30},
        'English': {'present': 29, 'total': 30},
        'Computer Science': {'present': 26, 'total': 30}
    },
    'jane_smith': {
        'Mathematics': {'present': 27, 'total': 30},
        'Physics': {'present': 28, 'total': 30},
        'Chemistry': {'present': 26, 'total': 30},
        'English': {'present': 29, 'total': 30},
        'Computer Science': {'present': 25, 'total': 30}
    },
    'bob_johnson': {
        'Mathematics': {'present': 29, 'total': 30},
        'Physics': {'present': 26, 'total': 30},
        'Chemistry': {'present': 28, 'total': 30},
        'English': {'present': 27, 'total': 30},
        'Computer Science': {'present': 29, 'total': 30}
    }
}

def load_teachers():
    """Load teachers from file"""
    global TEACHERS
    teachers_file = 'teachers.json'
    if os.path.exists(teachers_file):
        try:
            with open(teachers_file, 'r') as f:
                loaded_teachers = json.load(f)
                TEACHERS.update(loaded_teachers)
                print(f"Loaded {len(loaded_teachers)} teachers from file")
        except Exception as e:
            print(f"Error loading teachers: {e}")

def save_teachers():
    """Save teachers to file"""
    teachers_file = 'teachers.json'
    try:
        with open(teachers_file, 'w') as f:
            json.dump(TEACHERS, f, indent=4)
        print("Teachers saved successfully")
    except Exception as e:
        print(f"Error saving teachers: {e}")

def load_students():
    """Load students from file"""
    global STUDENTS
    students_file = 'students.json'
    if os.path.exists(students_file):
        try:
            with open(students_file, 'r') as f:
                loaded_students = json.load(f)
                STUDENTS.update(loaded_students)
                print(f"Loaded {len(loaded_students)} students from file")
        except Exception as e:
            print(f"Error loading students: {e}")

def save_students():
    """Save students to file"""
    students_file = 'students.json'
    try:
        with open(students_file, 'w') as f:
            json.dump(STUDENTS, f, indent=4)
        print("Students saved successfully")
    except Exception as e:
        print(f"Error saving students: {e}")

def load_attendance():
    """Load attendance records from file"""
    global STUDENT_ATTENDANCE
    attendance_file = 'student_attendance.json'
    if os.path.exists(attendance_file):
        try:
            with open(attendance_file, 'r') as f:
                loaded_attendance = json.load(f)
                STUDENT_ATTENDANCE.update(loaded_attendance)
                print(f"Loaded attendance records")
        except Exception as e:
            print(f"Error loading attendance: {e}")

def save_attendance():
    """Save attendance records to file"""
    attendance_file = 'student_attendance.json'
    try:
        with open(attendance_file, 'w') as f:
            json.dump(STUDENT_ATTENDANCE, f, indent=4)
        print("Attendance saved successfully")
    except Exception as e:
        print(f"Error saving attendance: {e}")

def load_known_faces():
    """Load all registered faces from the known_faces directory"""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir('known_faces'):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join('known_faces', filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
    
    print(f"Loaded {len(known_face_names)} known faces")

def get_all_registered_students():
    """Get all students who have registered faces"""
    registered_students = {}
    
    for filename in os.listdir('known_faces'):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            student_id = os.path.splitext(filename)[0]
            
            if student_id in STUDENTS:
                registered_students[student_id] = STUDENTS[student_id]
            else:
                registered_students[student_id] = {
                    'name': student_id.replace('_', ' ').title(),
                    'usn': 'N/A',
                    'password': 'student123',
                    'security_question': SECURITY_QUESTIONS[0],
                    'security_answer': 'default'
                }
    
    return registered_students

def get_all_attendance_dates():
    """Get all dates where attendance was recorded"""
    dates = []
    for filename in os.listdir('attendance_records'):
        if filename.startswith('attendance_') and filename.endswith('.json'):
            date_str = filename.replace('attendance_', '').replace('.json', '')
            dates.append(date_str)
    return sorted(dates)

def get_absent_dates_for_student(student_id, subject):
    """Get all dates when a student was absent for a specific subject"""
    absent_dates = []
    all_dates = get_all_attendance_dates()
    
    for date in all_dates:
        attendance_file = f'attendance_records/attendance_{date}.json'
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                attendance_data = json.load(f)
                
            # Check if student was absent (not in attendance or subject not marked)
            if student_id not in attendance_data:
                absent_dates.append(date)
            elif isinstance(attendance_data[student_id], dict):
                if subject not in attendance_data[student_id]:
                    absent_dates.append(date)
    
    return absent_dates

def mark_attendance(name, subject):
    """Mark attendance for a recognized person in a specific subject"""
    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    attendance_file = f'attendance_records/attendance_{today}.json'
    attendance_data = {}
    
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            attendance_data = json.load(f)
    
    if name not in attendance_data or not isinstance(attendance_data[name], dict):
        attendance_data[name] = {}
    
    if subject in attendance_data[name]:
        return False, attendance_data[name][subject]
    
    attendance_data[name][subject] = current_time
    
    with open(attendance_file, 'w') as f:
        json.dump(attendance_data, f, indent=4)
    
    if name not in STUDENT_ATTENDANCE:
        STUDENT_ATTENDANCE[name] = {}
    
    if subject not in STUDENT_ATTENDANCE[name]:
        STUDENT_ATTENDANCE[name][subject] = {'present': 0, 'total': 0}
    
    STUDENT_ATTENDANCE[name][subject]['present'] += 1
    STUDENT_ATTENDANCE[name][subject]['total'] += 1
    
    save_attendance()
    
    return True, current_time

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/portal')
def portal():
    return render_template('portal.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/reset-password')
def reset_password_page():
    return render_template('password.html')

# API Routes
@app.route('/api/get_subjects')
def get_subjects():
    return jsonify({'success': True, 'subjects': SUBJECTS})

@app.route('/api/get_security_questions')
def get_security_questions():
    return jsonify({'success': True, 'questions': SECURITY_QUESTIONS})

@app.route('/api/register_face', methods=['POST'])
def register_face():
    """Register a new face for student with password and security question"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        usn = data.get('usn', '').strip().upper()
        user_type = data.get('user_type', 'student')
        password = data.get('password', '').strip()
        security_question = data.get('security_question', '').strip()
        security_answer = data.get('security_answer', '').strip().lower()
        image_data = data.get('image')
        
        if not name or not image_data:
            return jsonify({'success': False, 'message': 'Name and image are required'})
        
        if user_type == 'student' and not usn:
            return jsonify({'success': False, 'message': 'USN is required for students'})
        
        if user_type == 'student' and not password:
            return jsonify({'success': False, 'message': 'Password is required for students'})
        
        if user_type == 'student' and len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'})
        
        if not security_question or not security_answer:
            return jsonify({'success': False, 'message': 'Security question and answer are required'})
        
        user_id = name.lower().replace(' ', '_')
        
        if user_type == 'student' and user_id in STUDENTS:
            return jsonify({'success': False, 'message': 'Student already registered'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face and encode
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if len(face_locations) == 0:
            return jsonify({'success': False, 'message': 'No face detected in the image'})
        
        if len(face_locations) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one face is visible'})
        
        # Save image
        image_path = os.path.join('known_faces', f'{user_id}.jpg')
        cv2.imwrite(image_path, image)
        
        # Add to database
        if user_type == 'student':
            STUDENTS[user_id] = {
                'name': name,
                'usn': usn,
                'password': password,
                'security_question': security_question,
                'security_answer': security_answer
            }
            save_students()
            
            STUDENT_ATTENDANCE[user_id] = {}
            for subject in SUBJECTS:
                STUDENT_ATTENDANCE[user_id][subject] = {'present': 0, 'total': 0}
            save_attendance()
        
        load_known_faces()
        
        return jsonify({'success': True, 'message': f'Successfully registered {name}'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/register_teacher', methods=['POST'])
def register_teacher():
    """Register a new teacher with subject and security question"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        password = data.get('password', '').strip()
        subject = data.get('subject', '').strip()
        security_question = data.get('security_question', '').strip()
        security_answer = data.get('security_answer', '').strip().lower()
        
        if not name or not password:
            return jsonify({'success': False, 'message': 'Name and password are required'})
        
        if not subject:
            return jsonify({'success': False, 'message': 'Subject is required'})
        
        if subject not in SUBJECTS:
            return jsonify({'success': False, 'message': 'Invalid subject selected'})
        
        if not security_question or not security_answer:
            return jsonify({'success': False, 'message': 'Security question and answer are required'})
        
        teacher_id = name.lower().replace(' ', '_')
        
        if teacher_id in TEACHERS:
            return jsonify({'success': False, 'message': 'Teacher already registered'})
        
        TEACHERS[teacher_id] = {
            'name': name,
            'password': password,
            'subject': subject,
            'security_question': security_question,
            'security_answer': security_answer
        }
        
        save_teachers()
        
        return jsonify({
            'success': True, 
            'message': f'Successfully registered teacher: {name} for {subject}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/api/recognize_face', methods=['POST'])
def recognize_face():
    """Recognize face and mark attendance for specific subject"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        subject = data.get('subject', '').strip()
        
        if not image_data:
            return jsonify({'success': False, 'message': 'Image data is required'})
        
        if not subject:
            return jsonify({'success': False, 'message': 'Subject is required'})
        
        if subject not in SUBJECTS:
            return jsonify({'success': False, 'message': 'Invalid subject selected'})
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        recognized_faces = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            top, right, bottom, left = face_location
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                    name = known_face_names[best_match_index]
                    
                    was_marked, attendance_time = mark_attendance(name, subject)
                    
                    student_name = STUDENTS.get(name, {}).get('name', name)
                    
                    recognized_faces.append({
                        'name': student_name,
                        'student_id': name,
                        'subject': subject,
                        'attendance_marked': was_marked,
                        'time': attendance_time,
                        'location': {
                            'top': int(top),
                            'right': int(right),
                            'bottom': int(bottom),
                            'left': int(left)
                        }
                    })
                else:
                    recognized_faces.append({
                        'name': 'Unknown',
                        'attendance_marked': False,
                        'location': {
                            'top': int(top),
                            'right': int(right),
                            'bottom': int(bottom),
                            'left': int(left)
                        }
                    })
            else:
                recognized_faces.append({
                    'name': 'Unknown',
                    'attendance_marked': False,
                    'location': {
                        'top': int(top),
                        'right': int(right),
                        'bottom': int(bottom),
                        'left': int(left)
                    }
                })
        
        if recognized_faces:
            return jsonify({'success': True, 'recognized': recognized_faces})
        else:
            return jsonify({'success': False, 'message': 'Face not recognized'})
    
    except Exception as e:
        print(f"Error in recognize_face: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_registered_users')
def get_registered_users():
    """Get list of all registered users"""
    try:
        users = list(get_all_registered_students().keys())
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_attendance')
def get_attendance():
    """Get today's attendance"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        attendance_file = f'attendance_records/attendance_{today}.json'
        
        attendance_data = {}
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                attendance_data = json.load(f)
        
        return jsonify({
            'success': True,
            'date': today,
            'attendance': attendance_data
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/student_login', methods=['POST'])
def student_login():
    """Student login"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip().lower().replace(' ', '_')
        usn = data.get('usn', '').strip().upper()
        password = data.get('password', '').strip()
        
        if name in STUDENTS:
            student = STUDENTS[name]
            if student['usn'] == usn and student['password'] == password:
                return jsonify({
                    'success': True,
                    'message': 'Login successful',
                    'student': {
                        'id': name,
                        'name': student['name'],
                        'usn': student['usn']
                    }
                })
        
        return jsonify({'success': False, 'message': 'Invalid credentials'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/teacher_login', methods=['POST'])
def teacher_login():
    """Teacher login"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip().lower().replace(' ', '_')
        password = data.get('password', '').strip()
        
        if name in TEACHERS:
            teacher = TEACHERS[name]
            if teacher['password'] == password:
                return jsonify({
                    'success': True,
                    'message': 'Login successful',
                    'teacher': {
                        'teacher_id': name,
                        'name': teacher['name'],
                        'subject': teacher['subject']
                    }
                })
        
        return jsonify({'success': False, 'message': 'Invalid credentials'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/student_attendance/<student_id>')
def get_student_attendance(student_id):
    """Get attendance for a specific student"""
    try:
        if student_id not in STUDENT_ATTENDANCE:
            return jsonify({'success': False, 'message': 'Student not found'})
        
        attendance = STUDENT_ATTENDANCE[student_id]
        subjects = []
        total_present = 0
        total_classes = 0
        
        for subject, data in attendance.items():
            present = data.get('present', 0)
            total = data.get('total', 0)
            percentage = (present / total * 100) if total > 0 else 0
            
            # Get absent dates for this subject
            absent_dates = get_absent_dates_for_student(student_id, subject)
            absent_count = len(absent_dates)
            
            subjects.append({
                'name': subject,
                'present': present,
                'total': total,
                'percentage': round(percentage, 1),
                'absent_count': absent_count,
                'absent_dates': absent_dates
            })
            
            total_present += present
            total_classes += total
        
        return jsonify({
            'success': True,
            'data': {
                'subjects': subjects,
                'total_present': total_present,
                'total_classes': total_classes
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/teacher_subject_attendance/<teacher_id>')
def get_teacher_subject_attendance(teacher_id):
    """Get attendance for teacher's subject"""
    try:
        if teacher_id not in TEACHERS:
            return jsonify({'success': False, 'message': 'Teacher not found'})
        
        subject = TEACHERS[teacher_id]['subject']
        today = datetime.now().strftime('%Y-%m-%d')
        attendance_file = f'attendance_records/attendance_{today}.json'
        
        today_attendance = {}
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                today_attendance = json.load(f)
        
        students = []
        for student_id in get_all_registered_students():
            if student_id in STUDENT_ATTENDANCE and subject in STUDENT_ATTENDANCE[student_id]:
                student_info = STUDENTS.get(student_id, {'name': student_id, 'usn': 'N/A'})
                data = STUDENT_ATTENDANCE[student_id][subject]
                
                present = data.get('present', 0)
                total = data.get('total', 0)
                percentage = (present / total * 100) if total > 0 else 0
                
                today_time = None
                if student_id in today_attendance and isinstance(today_attendance[student_id], dict):
                    today_time = today_attendance[student_id].get(subject)
                
                # Get absent dates for this subject
                absent_dates = get_absent_dates_for_student(student_id, subject)
                absent_count = len(absent_dates)
                
                students.append({
                    'id': student_id,
                    'name': student_info['name'],
                    'usn': student_info['usn'],
                    'present': present,
                    'total': total,
                    'percentage': round(percentage, 1),
                    'today_time': today_time,
                    'absent_count': absent_count,
                    'absent_dates': absent_dates
                })
        
        present_today = sum(1 for s in students if s['today_time'] is not None)
        
        return jsonify({
            'success': True,
            'data': {
                'subject': subject,
                'students': students,
                'stats': {
                    'total_students': len(students),
                    'present_today': present_today
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/delete_student/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    """Delete a student and all their data"""
    try:
        # Delete face image
        image_path = os.path.join('known_faces', f'{student_id}.jpg')
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Remove from STUDENTS
        if student_id in STUDENTS:
            del STUDENTS[student_id]
            save_students()
        
        # Remove from STUDENT_ATTENDANCE
        if student_id in STUDENT_ATTENDANCE:
            del STUDENT_ATTENDANCE[student_id]
            save_attendance()
        
        # Reload faces
        load_known_faces()
        
        return jsonify({
            'success': True,
            'message': f'Successfully deleted student: {student_id}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/verify_user', methods=['POST'])
def verify_user():
    """Verify user for password reset"""
    try:
        data = request.get_json()
        user_type = data.get('user_type')
        identifier = data.get('identifier', '').strip().lower().replace(' ', '_')
        usn = data.get('usn', '').strip().upper() if data.get('usn') else None
        
        if user_type == 'student':
            if identifier in STUDENTS:
                student = STUDENTS[identifier]
                if usn and student['usn'] != usn:
                    return jsonify({'success': False, 'message': 'USN does not match'})
                
                return jsonify({
                    'success': True,
                    'name': student['name'],
                    'security_question': student['security_question']
                })
        elif user_type == 'teacher':
            if identifier in TEACHERS:
                teacher = TEACHERS[identifier]
                return jsonify({
                    'success': True,
                    'name': teacher['name'],
                    'security_question': teacher['security_question']
                })
        
        return jsonify({'success': False, 'message': 'User not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/verify_security_answer', methods=['POST'])
def verify_security_answer():
    """Verify security answer and reset password"""
    try:
        data = request.get_json()
        user_type = data.get('user_type')
        identifier = data.get('identifier', '').strip().lower().replace(' ', '_')
        security_answer = data.get('security_answer', '').strip().lower()
        new_password = data.get('new_password', '').strip()
        
        if len(new_password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'})
        
        if user_type == 'student':
            if identifier in STUDENTS:
                student = STUDENTS[identifier]
                if student['security_answer'] == security_answer:
                    STUDENTS[identifier]['password'] = new_password
                    save_students()
                    return jsonify({'success': True, 'message': 'Password reset successful'})
        elif user_type == 'teacher':
            if identifier in TEACHERS:
                teacher = TEACHERS[identifier]
                if teacher['security_answer'] == security_answer:
                    TEACHERS[identifier]['password'] = new_password
                    save_teachers()
                    return jsonify({'success': True, 'message': 'Password reset successful'})
        
        return jsonify({'success': False, 'message': 'Incorrect security answer'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    load_known_faces()
    load_teachers()
    load_students()
    load_attendance()
    app.run(debug=True, host='0.0.0.0', port=5000)