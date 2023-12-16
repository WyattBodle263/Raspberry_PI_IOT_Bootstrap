# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)


def draw_pose(image, landmarks):
    # Copy the image
    landmark_image = image.copy()

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Create an instance of the holistic class
    holistic = mp.solutions.holistic.Holistic()

    results = holistic.process(landmark_image)

    color = (255, 0, 0)

    connectionPoints = []
    if results.pose_landmarks is not None:
        # For each landmark, get its x, y, z position
        for thisLandmark in results.pose_landmarks.landmark:
            x = int(thisLandmark.x * width)
            y = int(thisLandmark.y * height)
            z = thisLandmark.z
            vis = thisLandmark.visibility

            connectionPoints.append((x, y))
            radius = 5
            cv2.circle(landmark_image, (x, y), radius, color, -1)

        # Draw pose lines if enough landmarks are present
        if len(connectionPoints) >= 33:
            eyes = [connectionPoints[8], connectionPoints[6], connectionPoints[5], connectionPoints[4], connectionPoints[0], connectionPoints[1], connectionPoints[2], connectionPoints[3], connectionPoints[7]]
            connectLine(landmark_image, eyes)

            mouth = [connectionPoints[9], connectionPoints[10]]
            connectLine(landmark_image, mouth)

            left_arm = [connectionPoints[12], connectionPoints[14], connectionPoints[16], connectionPoints[18], connectionPoints[20], connectionPoints[16], connectionPoints[22]]
            connectLine(landmark_image, left_arm)

            right_arm = [connectionPoints[11], connectionPoints[13], connectionPoints[15], connectionPoints[17], connectionPoints[19], connectionPoints[15], connectionPoints[21]]
            connectLine(landmark_image, right_arm)

            torso = [connectionPoints[11], connectionPoints[12], connectionPoints[24], connectionPoints[23], connectionPoints[11]]
            connectLine(landmark_image, torso)

            left_leg = [connectionPoints[24], connectionPoints[26], connectionPoints[28], connectionPoints[30], connectionPoints[32], connectionPoints[28]]
            connectLine(landmark_image, left_leg)

            right_leg = [connectionPoints[23], connectionPoints[25], connectionPoints[27], connectionPoints[29], connectionPoints[31], connectionPoints[27]]
            connectLine(landmark_image, right_leg)

    return landmark_image

def connectLine(landmark_image, connectionPoints):
    # Connect the Lines
    for i in range(1, len(connectionPoints)):
        point_1 = connectionPoints[i]
        point_2 = connectionPoints[i - 1]

        x1, y1 = point_1
        x2, y2 = point_2

        # Make line
        cv2.line(landmark_image, (x1, y1), (x2, y2), (0, 0, 255), 2)


def main():
    while True:
        # Create a pose estimation model
        mp_pose = mp.solutions.pose

        # start detecting the poses
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

            # load image from pi camera
            image = pi_camera.capture_array()

            image.flags.writeable = False

            # get the landmarks
            results = pose.process(image)

            if results.pose_landmarks != None:
                result_image = draw_pose(image, results.pose_landmarks)
                cv2.imshow("Video", image)
                print(results.pose_landmarks)
            else:
                print('No Pose Detected')

            # Waits for 1ms and if the 1 is pressed it breaks the loop
            if cv2.waitKey(1) == ord('q'):
                break


# close all the windows
cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print('done')