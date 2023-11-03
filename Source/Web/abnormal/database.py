import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
from datetime import datetime
import time

class Database():
    def __init__(self, host_name, port, user_name, user_password):
        self.host_name = host_name
        self.port = port
        self.user_name = user_name
        self.user_password = user_password
        self.connection = None
        self.db_name = None
        self.time_in = None
        self.time_out = None

    #Connecting to sever----------------------------------------------------------
    def create_server_connection(self):
        try:
            self.connection=mysql.connector.connect(
                host = self.host_name,
                user = self.user_name,
                password = self.user_password
            )
            print("Mysql Database connection successful")
        except Error as err:
            print("Error:", err)

    
    #Creat Database-------------------------------------------------------------------
    def create_database(self):
        query = "Create database FPT_Project"
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            print("Databse created succcessfully!")
            self.db_name = "FPT_Project"
        except Error as err:
            print("Error:", err)
            
    #Connect to database-------------------------------------------------------
    def create_db_connection(self):
        try:
            self.connection = mysql.connector.connect(
                host = self.host_name,
                user = self.user_name,
                password = self.user_password,
                database = self.db_name
            )
            print(f"Connect to {self.db_name} successfully!")
        except Error as err:
            print(f"Error: {err}!")
    
    #Execute query:-------------------------------------------------------------
    def execute_query(self, query):
        cusor = self.connection.cursor()
        try:
            cusor.execute(query)
            self.connection.commit()
            print("Querry was successful!")
        except Error as err:
            print(f"Error: {err}!")

    #Creat all the table------------------------------------------------------------
    def creat_table_default(self):        
        #Students information
        create_students_info = """
        create table students_info(
            student_id varchar(10) primary key,
            student_name varchar(30) not null,
            student_class varchar(10) not null
            )
        """
        self.execute_query(create_students_info)
                
        #Time the students come in and get out of the exam room
        create_time_record = """
        create table time_record(
            serial int primary key,
            student_id varchar(10) not null,
            time_in datetime not null,
            time_out datetime not null,
            FOREIGN KEY (student_id) REFERENCES students_info(student_id) 
            )
        """
        self.execute_query(create_time_record)
        

        #subjects information
        create_time_subject = """
        create table time_subject(
            subject_code varchar(10) primary key,
            exam_datetime datetime not null
            )
        """
        self.execute_query(create_time_subject)
        
        #Exam rooms and subject of each room information
        create_exam_room = """
        create table exam_room(
            room_num int not null,
            subject_code varchar(10) not null,
            list_code int not null,
            primary key(list_code),
            FOREIGN KEY (subject_code) REFERENCES time_subject(subject_code)
            )
        """
        self.execute_query(create_exam_room)
        
        
        #Exam rooms and students in each room information
        create_room_info = """
        create table room_info(
            list_code int not null,
            student_id varchar(10) not null,
            FOREIGN KEY (list_code) REFERENCES exam_room(list_code),
            FOREIGN KEY (student_id) REFERENCES students_info(student_id) 
            )
        """
        self.execute_query(create_room_info)
    
        
    #Insert default data to the table------------------------------------------------------------------------
    def insert_default(self):
        #Students information
        students_info = """
        insert into students_info values
        ("QE170110", "NguyenThanhDat", "AI17B"),
        ("QE170066", "NgoDinhAn", "AI17B"),
        ("QE170069", "TruongTrongTien", "AI17B"),
        ("QE170078", "PhamQuocHung", "AI17B"),
        ("QE170017", "HoTonBao", "AI17B"),
        ("QE170053", "TranTienDat", "AI17B"),
        ("QE170013", "HoangThanhLam","AI17B"),
        ("QE170008", "TruongQuyetThang", "AI17B"),
        ("QE170156", "DangThiLeChi", "AI17B"),
        ("QE170004", "MaiXuanHuu", "AI17B"),
        ("QE170005", "TranThongNhat", "AI17B"),
        ("QE170006", "LeHuyHoan", "AI17B"),
        ("QE170049", "DiepGiDong", "AI17B"),
        ("QE170085", "PhanQuocTrung", "AI17B"),
        ("QE170090", "TruongPhuocTrung", "AI17B"),
        ("QE170105", "DuongThanhDuy", "AI17B"),
        ("QE170224", "NguyenTanKiet", "AI17B");
        """
        self.execute_query(students_info)
        
        #Subjects information
        time_subject = """
        insert into time_subject values
        ("MAD101", "2023-05-04 14:00:00"),
        ("JPD113", "2023-05-04 14:00:00"),
        ("MAE101", "2023-05-06 10:00:00");
        """
        self.execute_query(time_subject)
        
        #Exam rooms information
        exam_room = """
        insert into exam_room values
        (406, "MAD101", 1),
        (407, "MAD101", 2),
        (408, "MAD101", 3),
        (409, "JPD113", 4),
        (410, "JPD113", 5),
        (411, "JPD113", 6),
        (406, "MAE101", 7),
        (407, "MAE101", 8),
        (408, "MAE101", 9);
        """
        self.execute_query(exam_room)
        
        #Exam room list
        room_info = """
        insert into room_info values
        (1, "QE170110"),
        (1, "QE170066"),
        (1, "QE170069"),
        (1, "QE170078"),
        (2, "QE170017"),
        (2, "QE170053"),
        (2, "QE170013"),
        (2, "QE170008"),
        (2, "QE170156"),
        (3, "QE170004"),
        (3, "QE170005"),
        (3, "QE170006"),
        (3, "QE170049"),
        (4, "QE170085"),
        (4, "QE170090"),
        (4, "QE170105"),
        (4, "QE170224");
        """
        self.execute_query(room_info)
        

        
        
        
        
    #Drop table-------------------------------------------------------
    def drop_table(self):
        table_name = input("Input name of table: ")
        drop = f"""
        drop table {table_name} 
        """
        #time_record là tên bảng
        self.execute_query(drop)
    
    
    #Creat table ------------------------------------------------------
    def creat_table(self):
        query = input("Input query: ")
        self.execute_query(query)
    
    #DROP DATABASE--------------------------------------
    def drop_database(self):
        query = '''drop database FPT_Project'''
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            print("Database dropped succcessfully!")
            self.db_name = None
        except Error as err:
            print("Error:", err)
    
    
    #insert data----------------------------------------
    def insert_data(self):
        query = input("Input query: ")
        self.execute_query(query)
        
    
                
    #Read query--------------------------------------------------------
    def read_query(self,query):
        cursor = self.connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            print("Read successfully!")
            return result
        except Error as err:
            print(f"Error: {err}!")
            
            
    #Read info of studnet follow id------------------------------------
    def read_info(self):
        id = input("Input Student ID: ")
        query = f"""
        select * from students_info where student_id = "{id}";
        """
        result = self.read_query(query)
        print(result)
    
    #Get the currently time_in-------------------------------------------
    def get_time_in(self):
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        self.time_in = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Successfully!!!")
        
    #Get the currently time_out-------------------------------------------
    def get_time_out(self):
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        self.time_out = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Successfully!!!")
    
    #Write time to database-------------------------------------------
    def write_time(self):
        id = input("Input student ID: ")
        #counting numbers of record
        count = """
        select count(*) from time_record;
        """
        result = self.read_query(count)[0][0]
        print(result)
        print(type(result))
        #wwiting time
        write_time_query = f"""
        insert into time_record values
        ({result+1},"{id}", "{self.time_in}", "{self.time_out}");
        """
        self.execute_query(write_time_query)
    
    
    #Get the students list of an exam room in a specific time
    def get_students_in_room(self):
        exam_time = input("Input the time you want to check (YYYY-MM-DD HH:MM:SS): ")
        query = f'''
        select * from time_subject where exam_datetime = "{exam_time}"'''
        result = pd.DataFrame(self.read_query(query))
        print(result)
        subject_code = input("Which subject do you want to check? ")
        query = f'''
        select * from exam_room where subject_code = "{subject_code}"'''
        result = pd.DataFrame(self.read_query(query))
        print(result)
        list_code = input("Which list do you want to get? ")
        query = f'''
        select * from room_info where list_code = "{list_code}"'''
        result = pd.DataFrame(self.read_query(query))
        print(result)
        result.to_csv("List_code.csv")
        print("Save file successfully!!!")
    
    
    
    
        
if __name__=="__main__":
    db = Database("127.0.0.1","3306","root","16052003daT@")
    db.create_server_connection()
    while True:
        print("""
Choose action:
    1. Create Dream Face database
    2. Drop Dream Face database
    3. Create table default
    4. Insert data default
    5. Drop table
    6. Insert data
    7. Read student information
    8. Get time in
    9. Get time out
    10.Write time
    11.Get student list in each room
    12. Exit
        """)
        try:
            action = int(input("Choose your action: "))
        except:
            print("Please input number 1-11!!!")
            continue
        if action == 1:
            db.create_database()
            db.create_db_connection()
        elif action == 2:
            db.drop_database()
        elif action ==3:
            db.creat_table_default()
        elif action == 4:
            db.insert_default()
        elif action == 5:
            db.drop_table()
        elif action == 6:
            db.insert_data()
        elif action == 7:
            db.read_info()
        elif action == 8:
            db.get_time_in()
        elif action == 9:
            db.get_time_out()
        elif action == 10:
            db.write_time()
        elif action == 11:
            db.get_students_in_room()
        elif action == 12:
            break
        else:
            print("Please input 1-12!!!")