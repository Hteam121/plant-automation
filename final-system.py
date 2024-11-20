import time
import datetime
import threading
from gpiozero import OutputDevice, InputDevice
from google.cloud import firestore
from google.oauth2 import service_account
from datetime import timedelta  # for adding 5 hours

# Initialize Firestore
credentials = service_account.Credentials.from_service_account_file("/home/temoc/Desktop/plant-automation/CREDS.json")
db = firestore.Client(credentials=credentials)

motor1 = OutputDevice(22)
motor2 = OutputDevice(27)
soil_sensor1 = InputDevice(17)  
soil_sensor2 = InputDevice(18)  
water_level_sensor = InputDevice(13)

# Sensor check interval is 10 seconds, checking every 10 seconds
CHECK_INTERVAL = 10  
DRY_THRESHOLD = 30  

# Function to record sensor data to Google Cloud Firestore
def record_sensor_data(sensor_name, value):
    timestamp = datetime.datetime.now() + timedelta(hours=5)  # Add 5 hours
    data = {
        'sensor': sensor_name,
        'value': value,
        'timestamp': timestamp
    }
    db.collection('sensor_logs').add(data)
    print(f"Logged {sensor_name}: {value}, Timestamp: {timestamp}")

# Function to record motor status to Firestore
def record_motor_status(motor_name, status):
    timestamp = datetime.datetime.now() + timedelta(hours=5)  # Adjust time if necessary
    data = {
        'motor': motor_name,
        'status': status,
        'timestamp': timestamp
    }
    db.collection('motors').document(motor_name).set(data)
    print(f"Logged {motor_name}: {status}")

# Function to monitor soil moisture and water level
def monitor_sensors():
    last_dry_time1 = None
    last_dry_time2 = None
    print("Monitoring sensors...")
    
    while True:
        now = time.time()

        # Check Soil Sensor 1
        sensor1_status = 'Dry' if soil_sensor1.is_active else 'Wet'
        record_sensor_data('Soil Sensor 1', sensor1_status)
        print(f"Soil Sensor 1: {sensor1_status}")
        
        if sensor1_status == 'Dry':
            if last_dry_time1 is None:
                last_dry_time1 = now
            elif now - last_dry_time1 > DRY_THRESHOLD:
                print("Soil Sensor 1 has been dry for over 30 seconds. Turning on Motor 1.")
                motor1.on()
                record_motor_status("motor1", "on")  # Log motor status as 'on'
                time.sleep(10)  # Run motor for 10 seconds
                motor1.off()
                record_motor_status("motor1", "off")  # Log motor status as 'off'
                last_dry_time1 = None  # Reset the dry time after turning on the motor
        else:
            last_dry_time1 = None  # Reset if it's not dry

        # Check Soil Sensor 2
        sensor2_status = 'Dry' if soil_sensor2.is_active else 'Wet'
        record_sensor_data('Soil Sensor 2', sensor2_status)
        print(f"Soil Sensor 2: {sensor2_status}")

        if sensor2_status == 'Dry':
            if last_dry_time2 is None:
                last_dry_time2 = now
            elif now - last_dry_time2 > DRY_THRESHOLD:
                print("Soil Sensor 2 has been dry for over 30 seconds. Turning on Motor 2.")
                motor2.on()
                record_motor_status("motor2", "on")  # Log motor status as 'on'
                time.sleep(10)  
                motor2.off()
                record_motor_status("motor2", "off")  # Log motor status as 'off'
                last_dry_time2 = None  # Reset the dry time after turning on the motor
        else:
            last_dry_time2 = None  

        # Check Water Level Sensor
        water_level_status = 'Low' if water_level_sensor.is_active else 'High'
        record_sensor_data('Water Level Sensor', water_level_status)
        print(f"Water Level Sensor: {water_level_status}")

        time.sleep(CHECK_INTERVAL)  # Check sensors every 10 seconds

# Function to listen for motor status changes from Firestore and control motors
def listen_for_motor_status():
    print("Listening for motor status changes...")

    def on_motor1_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            status = doc.get('status')
            print(f"Motor 1 status changed to {status}")
            if status == 'on':
                motor1.on()
                time.sleep(10)  # Wait 10 seconds after motor is manually turned on
                motor1.off()

    def on_motor2_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            status = doc.get('status')
            print(f"Motor 2 status changed to {status}")
            if status == 'on':
                motor2.on()
                time.sleep(10)  # Wait 10 seconds after motor is manually turned on
                motor2.off()

    # Watch the 'motors' collection documents for motor 1 and motor 2
    doc_ref_motor1 = db.collection('motors').document('motor1')
    doc_watch_motor1 = doc_ref_motor1.on_snapshot(on_motor1_snapshot)

    doc_ref_motor2 = db.collection('motors').document('motor2')
    doc_watch_motor2 = doc_ref_motor2.on_snapshot(on_motor2_snapshot)

    while True:
        time.sleep(1)  # to keep the script running

# Create threads for both the monitoring and listening functions
try:
    # Run both monitor_sensors and listen_for_motor_status in parallel using threading
    threading.Thread(target=listen_for_motor_status).start()
    threading.Thread(target=monitor_sensors).start()

except KeyboardInterrupt:
    print("Exiting and turning off all motors.")
    motor1.off()
    motor2.off()
