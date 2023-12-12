import datetime
import os
import time
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from tqdm import tqdm
import cv2

# TODO: Make filepaths absolute

def test_powerup():
    # first, test that GMAIL_EMAIL and GMAIL_KEY are set in env
    email_env = os.environ["GMAIL_EMAIL"]
    email_key_env = os.environ["GMAIL_KEY"]
    # check that both are set
    if "GMAIL_EMAIL" in os.environ and "GMAIL_KEY" in os.environ:
        # test if we can log into gmail
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(os.environ["GMAIL_EMAIL"], os.environ["GMAIL_KEY"])
            print("Valid Email detected! Using email: " + os.environ["GMAIL_EMAIL"])
        except:
            print('Incorrect gmail info')
            print('GMAIL_EMAIL: ' + os.environ["GMAIL_EMAIL"])
            print('GMAIL_KEY: ' + os.environ["GMAIL_KEY"])
            exit(1)

    else:
        print('GMAIL_EMAIL and GMAIL_KEY must be set in env')
        print('GMAIL_EMAIL: ' + os.environ["GMAIL_EMAIL"])
        print('GMAIL_KEY: ' + os.environ["GMAIL_KEY"])
        exit(1)





def capture(detect_cycle=False, real_cycle=False):
    current_time = datetime.datetime.now()

    # Get the width and height of the image
    width = 10000
    height = 10000

    # Get the filepath to save the image to
    filepath = "/home/evin/github/pisentry/test.jpg"

    assert width > 0 and height > 0, "Width and height must be positive"
    assert type(width) == int and type(height) == int, "Width and height must be integers"
    assert type(filepath) == str, "Filepath must be a string"

    # os_command = ["libcamera-still", "--width {} --height {} -o {}".format(width, height, filepath)]

    # print(os_command)

    # Run the test suite
    # subprocess.run(os_command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True, timeout=20)

    os.system("libcamera-still -o /home/evin/github/pisentry/test.jpg --vflip --hflip")

    # return

    while True:
        time.sleep(5)
        detect_person(image_path=filepath, detect_cycle=detect_cycle, real_cycle=real_cycle)
        break
            # break

def detect_person(image_path, detect_cycle=False, real_cycle=False):
    print("run person detection on the image")
    # read the filepath
    image = cv2.imread(image_path)

    # print(image)
    prototxt = "/home/evin/github/pisentry/MobileNetSSD_deploy.prototxt.txt"
    model = "/home/evin/github/pisentry/MobileNetSSD_deploy.caffemodel"

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
		0.007843, (300, 300), 127.5)
    
    # run the image through the network
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        prob = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        # min_confidence = 0.2
        # if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
 
        box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])

        # loop over the detections
        box_name = []
        box_conf = []
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.1:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                
                box_name.append(CLASSES[idx])
                box_conf.append(confidence * 100)

                cv2.rectangle(image, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
    
    # write the image with rectangle to the filepath, but with a different name
    cv2.imwrite("/home/evin/github/pisentry/test-cv.jpg", image)
    time.sleep(5)

    if real_cycle:
        send_email("PiCam1 Routine Photo", "Here it is", "evin@wustl.edu", attachment="/home/evin/github/pisentry/test-cv.jpg")
    elif detect_cycle:
        # this is a cycle that runs every 5 minutes, so we should only send an email if a person is detected
        if "person" in box_name:
            things = ""
            for i in set(box_name):
                things += i + ", "
            body = "Person Detected at Evin's Apartment\n" + things + "\n" + str(box_conf)
            send_email("{} Detected at Evin's Apartment".format(things), body, "evin@wustl.edu", attachment="/home/evin/github/pisentry/test-cv.jpg")


    pass

def send_email(subject, body, to, attachment=None):
    msg = MIMEMultipart()
    msg['From'] = os.environ["GMAIL_EMAIL"]
    msg['To'] = to
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Process the attachment, if there is one
    if attachment:
        try:
            with open(attachment, 'rb') as file:
                img = MIMEImage(file.read())
                img.add_header('Content-Disposition', 'attachment', filename=attachment)
                msg.attach(img)
        except IOError:
            print(f"Could not read the file {attachment}")

    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(os.environ["GMAIL_EMAIL"], os.environ["GMAIL_KEY"])
    print("Email sent to " + to)
    server.sendmail(os.environ["GMAIL_EMAIL"], to, text)
    server.quit()
   
    return True




if __name__ == "__main__":
    cycles = 0
    test_powerup()
    time.sleep(5)
    while True:
        capture(detect_cycle=True, real_cycle=False)
        print("sleeping until next cycle...")
        for i in tqdm(range(100)):
            time.sleep(0.1)

            

        cycles += 1
        if cycles == 2000:
            print("taking a real photo")
            capture(real_cycle=True)
            cycles = 0
        print("cycles: " + str(cycles))
        time.sleep(5)

