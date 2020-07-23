"""
python3 main.py -l 0 -fd_path models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
-hpe_path models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 \
-fld_path models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 \
-gd_path models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 \
-fd_th 0.3 -i bin/img.png
"""

#from src.input_feeder import InputFeeder
from src.face_detection import face_detection
from src.head_pose_estimation import head_pose_estimation
from src.facial_landmarks_detection import facial_landmarks_detection
from src.gaze_detection import gaze_detection
from src.mouse_controller import MouseController

from argparse import ArgumentParser
import cv2
import logging as log

#log.basicConfig(format='[INFO] \t %(message)s', level=log.INFO)

def print_stats(log_flag, banner, fd_time, hpe_time, fld_time, gd_time):
    if(log_flag==1):
        print(banner)
        log.info(f"FACE DETECTION:              {round(fd_time, 4)}s")
        log.info(f"HEAD POSE ESTIMATION:        {round(hpe_time, 4)}s")
        log.info(f"FACIAL LANDMARKS DETECTION:  {round(fld_time, 4)}s")
        log.info(f"GAZE DETECTION:              {round(gd_time, 4)}s")
        print("==============================================================\n")

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-l","--log_flag",type=int,default= 1, help="True: Log | False: No Logs")
    parser.add_argument("-fd_path", "--face_detection_model_path", required=True, type=str,
                        help="Path to an xml file face_detection model.")
    parser.add_argument("-hpe_path", "--head_pose_estimation_model_path", required=True, type=str,
                        help="Path to an xml file head_pose_estimation model.")
    parser.add_argument("-fld_path", "--facial_landmarks_detection_model_path", required=True, type=str,
                        help="Path to an xml file facial_landmarks_detection model.")
    parser.add_argument("-gd_path", "--gaze_detection_model_path", required=True, type=str,
                        help="Path to an xml file gaze_detection_detection model.")
    parser.add_argument("-fd_th", "--face_detection_threshold", type=float, default=0.5,
                        help="threshold for face_detection model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or '0' for webcam")
    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Target Device to infer on. CPU by default")
    return parser

if __name__ == '__main__':
    print("\n\n\t\t\t***PREFACE***")
    log.info('To diable all logs, comment out \n\t `log.basicConfig(format=`[INFO] %(message)`s, level=log.INFO)` from every module')
    log.info("To enable debugging, change log level from `level=log.INFO` to `level=log.DEBUG`\n")
    args = build_argparser().parse_args()

    # Initialise All Class instances
    fd = face_detection(args.face_detection_model_path)
    hpe = head_pose_estimation(args.head_pose_estimation_model_path)
    fld = facial_landmarks_detection(args.facial_landmarks_detection_model_path)
    gd = gaze_detection(args.gaze_detection_model_path)
    mc = MouseController('high', 'fast')

    ## Model 1
    fd_time = fd.load_model()
    fd.check_model()
    ## Model 2
    hpe_time = hpe.load_model()
    hpe.check_model()
    ## Model 3
    fld_time = fld.load_model()
    fld.check_model()
    ## Model 4
    gd_time = gd.load_model()
    gd.check_model()
    print_stats(args.log_flag,"======================== Loading Models ======================", fd_time, hpe_time, fld_time, gd_time)

    # Bring in Frames
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out.avi', fourcc, 20.0, (232,351))

    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        pframe = fd.preprocess_input(frame)
        fd_time = fd.predict(pframe)                # Model 1
        if(fd.wait() == 0):
            boxed_faces, fd_frame = fd.preprocess_output(frame, args.face_detection_threshold)
            # Infer on detected faces
            for bf in boxed_faces:
                # Using only the coordinates of the boxes and not the image
                face = frame[bf[1]:bf[3],bf[0]:bf[2]]
                pframe = hpe.preprocess_input(face)
                hpe_time = hpe.predict(pframe)       # Model 2
                yaw, pitch, roll = 0,0,0
                if(hpe.wait() == 0):
                    yaw, pitch, roll = hpe.preprocess_output()

                # Using the faces detected by Model 1
                pframe = fld.preprocess_input(face)
                fld_time = fld.predict(pframe)       # Model 3
                if(fld.wait() == 0):
                    left_eye_box, right_eye_box, face_with_landmarks, left_eye_center, right_eye_center = fld.preprocess_output(face)
                    #print(five_coord_pairs)

                    input_dict = gd.preprocess_input(face, left_eye_box, right_eye_box, [yaw, pitch, roll])
                    gd_time = gd.predict(input_dict) # Model 4
                    if(gd.wait() == 0):
                        face_with_gaze, x,y,z = gd.preprocess_output(face_with_landmarks, left_eye_center, right_eye_center)

                        # Now Move the Mouse Pointer
                        mc.move(x,y)       # Model 5
                    face_with_gaze = cv2.resize(face_with_gaze, (232,351))
                    cv2.imshow("res.png", face_with_gaze)
                    cv2.waitKey(150)
                    #cv2.imwrite("res"+str(frame_count)+".png",face_with_gaze)
                    out.write(face_with_gaze)

                    print_stats(args.log_flag,"=============== Avg. Inference Time Per Frame ================",fd_time, hpe_time, fld_time, gd_time)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
