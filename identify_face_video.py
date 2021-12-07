from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import socket
from datetime import datetime
import base64
import pymysql

from VideoGet import VideoGet
from threading import Thread

modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
video_url = 'rtsp://admin:miraicam2019@192.168.0.229:554/live'
#video_url = './tmp (3).mp4'
#video_url = 0

TCP_IP = '162.17.0.74'
#TCP_IP = '100.100.100.11'
TCP_PORT = 13522 

list_videoProcessor = []

database_host = "127.0.0.1"
database_port = 3306
database_user = 'root'
database_passwd = ''
database_name = 'faceServer'
database_charset = 'utf8'

with tf.Graph().as_default():
    #tf_config = tf.ConfigProto(device_count = {'GPU': 0})
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf_config.allow_soft_placement = True

    sess = tf.Session(config=tf_config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 180  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        image_size = 182
        input_image_size = 160

        print('Loading Modal')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')

        print('Start Recognition')

class VideoProcess :
    def __init__(self, dataset_folder="dataset" , ip = '162.17.0.74' , port = 13522 , index = 0 , video_url = '0' , returnType = 1):
        videoGetter = VideoGet(video_url).start()
        self.video_url = video_url
        self.frame = videoGetter.frame
        self.getter = videoGetter
        self.index = index
        self.returnType = returnType
        self.dataset_folder = dataset_folder
        self.detectedId = "-1_NoUser"
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.randName = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.snapFolderName = self.dataset_folder + "/" + self.randName + "_cam_" + str(self.index) + "/"
        if not os.path.exists(self.snapFolderName):
            os.makedirs(self.snapFolderName)
        self.connectServer()
        self.countSnap = 0
    def connectServer(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
    def start(self):
        Thread(target=self.senddata , args=()).start()
        Thread(target=self.process , args=()).start()
        return self
    
    def senddata(self):
        try:
            while True:
                if self.getter.stopped or self.getter.frame is None:
                    videoGetter = VideoGet(self.video_url)
                    self.getter = videoGetter
                    self.getter.start()
                    continue
                
                #frame = self.process()
                frame = self.frame
                frame = cv2.resize(src=frame, dsize=(0, 0), fx=0.5, fy=0.5)

                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),50]
                result, imgencode = cv2.imencode('.jpg', frame, encode_param)
                data = np.array(imgencode)

                stringData = base64.b64encode(data)
                length = str(len(stringData))

                if len(stringData) > 65000:
                    print("buffer is big")
                    continue
                if (self.returnType == 1):
                    self.sock.sendall(length.encode('utf-8').ljust(64))
                    self.sock.send(stringData)
                self.sock.send(self.detectedId.encode('utf-8').ljust(64))
                print("========================================================" + self.detectedId)

                self.detectedId = "-1_NoUser"
                self.sended = 1

                time.sleep(0.066)

        except Exception as e:
            print(e)

    def safe_snap(self , frame):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!taked!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        index = str(int(self.countSnap)).zfill(10)
        cv2.imwrite(self.snapFolderName + self.randName + "_" + str(index) + ".jpg",frame)
        self.countSnap = self.countSnap + 1

    def process(self):
        while True:
            if self.getter.stopped or self.getter.frame is None:
                videoGetter = VideoGet(self.video_url)
                self.getter = videoGetter
                self.getter.start()
                continue

            frame = self.getter.frame

            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print('Detected_FaceNum: %d' % nrof_faces)
            if self.sended != 1:
                continue
            if nrof_faces > 0 and self.sended == 1:
                # self.detectedId = ''
                self.sended = 0
                det = bounding_boxes[:, 0:4]
                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces,4), dtype=np.int32)
                for i in range(nrof_faces) :
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    if bb[i][0] < 0:
                        bb[i][0] = 0
                    if bb[i][1] < 0:
                        bb[i][1] = 0
                    if bb[i][2] > len(frame[0]):
                        bb[i][2] = len(frame[0])
                    if bb[i][3] > len(frame):
                        bb[i][3] = len(frame)

                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                        interpolation=cv2.INTER_CUBIC)

                    self.safe_snap(scaled[i])
                    
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    print(best_class_indices,' with accuracy ',best_class_probabilities)
                    if best_class_probabilities > 0.1 and best_class_probabilities < 0.2:
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 5)
                    elif best_class_probabilities > 0.2:
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 5)
                        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" + class_names[best_class_indices[0]])
                        if self.detectedId == '':
                            self.detectedId = class_names[best_class_indices[0]]
                        else :
                            self.detectedId = self.detectedId + "," + class_names[best_class_indices[0]]
                #return frame
                self.frame = frame
            else :
                #return frame
                self.frame = frame
    
def main():
    #videoGetter = VideoGet(video_url).start()
    index = 0
    try:
        connection = pymysql.connect(host = database_host , user = database_user , 
                                        port = database_port , passwd = database_passwd , 
                                        db = database_name , charset = database_charset)
        sql = 'SELECT tb_clientRoles.port , tb_clients.IP_address , tb_cameras.url , tb_clientRoles.returnType FROM tb_clientRoles \
                    LEFT JOIN tb_clients ON tb_clientRoles.client_id = tb_clients.id \
                    LEFT JOIN tb_cameras ON tb_clientRoles.camera_id = tb_cameras.id\
                    WHERE tb_clientRoles.is_delete = 0 AND tb_clientRoles.is_active = 1'

        cur=connection.cursor()
        cur.execute(sql)

        data=cur.fetchall()

        for item in data:
            print("port:" + str(item[0]) + ", ip:" + str(item[1]) + " , url : " + str(item[2]) + " , returnType : " + str(item[3]))
            videoProcessor = VideoProcess(dataset_folder="dataset" , ip = str(item[1]) , port = item[0] , index = index , video_url = str(item[2]) , returnType = item[3])
            list_videoProcessor.append(videoProcessor.start())
            index = index + 1
        cur.close()
        connection.close()
    except Exception:print("error occur")


    #videoProcessor = VideoProcess(dataset_folder="dataset" , ip = TCP_IP, port = TCP_PORT , index = 0 , video_url = video_url).start()

if __name__ == "__main__":
    main()