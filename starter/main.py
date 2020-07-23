"""
python3 main.py -fd_path model models/intel/face_detection-adas-binay-0001/FP32-INT1/face-detection-adas-binary-0001 -it image -ip bin/img.png
"""

from src.input_feeder import InputFeeder
from src.face_detection import face_detection

from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd_path", "--face_detection_model_path", required=True, type=str,
                        help="Path to an xml file face_detection model.")
    parser.add_argument("-fd_th", "--face_detection_threshold", type=float, default=0.5,
                        help="threshold for face_detection model")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Type of Input | 'video' : video Feed | 'image' : Image Feed | 'cam' : Webcam Feed")
    parser.add_argument("-ip", "--input_path", type=str,
                        help="Path to image or video file")
    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Target Device to infer on. CPU by default")
    return parser

if __init__ == '__main__':
    args = build_argparser().parse_args()
    # Load All Models
    ## Model 1
    fd = face_detection(args.face_detection_model_path)
    fd.load_model()
    fd.check_model()
    ## Model 2

    # Bring in Frames
    feed = InputFeeder(input_type=args.input_type, input_file=args.input_path)
    feed.load_data()
    for batch in feed.next_batch():
        pframe = fd.preprocess_input(feed)
        result = fd.predict(pframe)
        objects = fd.preprocess_output(result)

        frame = cv2.imread(batch)
        for obj in objects:
            if(obj[2] < args.face_detection_threshold): continue
            cv2.rectangle(batch,
                        (obj[3], obj[4]), (obj[5], obj[6]),
                        (0,0,0))

        cv2.imwrite("res.png",batch)
        cv2.imshow("res", batch)

    feed.close()
