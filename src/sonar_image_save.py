#!/usr/bin/env python3
import rospy
import rospkg
from sensor_msgs.msg import Image
import numpy as np
import cv2
import random
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

CLASSES = {"background": 0, "bottle": 1, "can": 2, "chain": 3, "drink-carton": 4,
            "hook": 5, "propeller": 6, "shampoo-bottle": 7, "standing-bottle": 8,
            "tire": 9, "valve": 10, "wall": 11}
MODELS = ['', 'bottle', 'Coke', 'chain', 'drink_carton',
          'hook', 'propeller', 'shampoo_bottle', 'standing_bottle',
          'car_wheel', 'valve', 'wall']
save_path = rospkg.RosPack().get_path('sonar_image_save')+'/sonar_imgs/'
th = 50
pitch = 15.0
bridge = CvBridge() # Initialize the CvBridge class
h = 0.7

def image_callback(img_msg):
    # log some info about the image topic
    rospy.loginfo(img_msg.header)
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, dsize=(320, 480))
        '''
        for x in range(320):
            for y in range(480):
                dist = np.sqrt((x-160)**2+y**2)
                cv_image[y][x] = min(cv_image[y][x]*(h/np.sin(np.deg2rad(dist/480*15+7.5))-1.5), 255)
        '''
        return cv_image
    except CvBridgeError:
          rospy.logerr("CvBridge Error: {0}".format(e))

def mask_callback(img_msg):
    # log some info about the image topic
    rospy.loginfo(img_msg.header)
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, dsize=(320, 480))
        return cv_image
    except CvBridgeError:
          rospy.logerr("CvBridge Error: {0}".format(e))

def spawn_model(model_name, x = 0, y = 0, z = 0, o_x = 0, o_y = 0, o_z = 0, o_w = 0):
    initial_pose = Pose()
    initial_pose.position.x = x
    initial_pose.position.y = y
    initial_pose.position.z = z

    initial_pose.orientation.x = o_x
    initial_pose.orientation.y = o_y
    initial_pose.orientation.z = o_z
    initial_pose.orientation.w = o_w

    # Spawn the new model #
    model_xml = ''
    with open (rospkg.RosPack().get_path('sonar_image_save')+'/models/'+model_name+'/model.sdf', 'r') as xml_file:
        model_xml = xml_file.read().replace('\n', '')

    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    spawn_model_prox(model_name, model_xml, '', initial_pose, 'world')

def delete_model(model_name):
    # Delete the old model if it's still around
    delete_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    delete_model_prox(model_name)

def get_mask(class_id):
    global j
    mask = np.zeros((480, 320), np.float32)
    for i in range(10):
        img_msg = rospy.wait_for_message("/blueview_p900/sonar_image", Image)
        src = mask_callback(img_msg)
        mask += src
    mask /= 10
    mask = mask.astype(np.uint8)
    ret, mask = cv2.threshold(mask, th, class_id, cv2.THRESH_BINARY)
    return mask

def main():
    # Initialize the ROS Node named 'sonar_image_save', allow multiple nodes to be run with this name
    rospy.init_node('sonar_image_save', anonymous=True)
    
    q = quaternion_from_euler(0, np.deg2rad(pitch), 0)
    spawn_model('blueview_p900_nps_multibeam_ray', 0, 0, 30, q[0], q[1], q[2], q[3]) # pitch 15 degree

    rospy.sleep(1.)  # delay

    i = 0
    j = 0
    # Multi
    while True:
        samples = random.randint(1,5)
        models = random.sample(['bottle', 'can', 'chain', 'drink-carton', 'hook', 'propeller', 'shampoo-bottle', 'standing-bottle', 'tire', 'valve', 'wall'], samples)  # except wall
        masks = []
        xyz = []

        rand_x = sorted(random.sample(range(1,6), samples), reverse=True)
        for k in range(samples):
            z = (h-0.1)-(random.random()*0.01)
            # pitch 15
            x = z/(0.23*(rand_x[k]/5.0)+0.15) + random.random()*0.1       #0.15~0.38 (h=0.7, 1.842~4.667)
            y = (x*0.23)*2*random.random() - (x*0.23)   # -0.23~0.3

            # pitch 30
            #x = z/(0.26*(rand_x[k]/5.0)+0.45)  # 0.4142~0.7673 -> 0.45 ~ 0.71
            #y = (x*0.23)*2*random.random() - (x*0.23)   # -0.23~0.3
            if models[k] == 'wall':
                xyz.append([5.4 + random.random()*0.1, random.random()*4-2, 30-h+2, 0, 0, 0, 0])
            elif models[k] == 'valve':
                xyz.append([x, y, 30-z+0.14, 0, 0, 0, 0])
            elif models[k] == 'standing-bottle':    # standing
                xyz.append([x, y, 30-z-0.05, 0.7071067, 0, 0, 0.7071069])
            elif models[k] == 'bottle':             # lying horizontally
                xyz.append([x, y, 30-z, 0, 0, 0, 0])
            elif models[k] == 'shampoo-bottle':     # vertical
                xyz.append([x, y, 30-z-0.05, 0.7071067, 0, 0, 0.7071069])
            elif models[k] == 'tire':               # horizontally
                xyz.append([x, y, 30-z-0.05, 0, 0, 0, 0])
            elif models[k] == 'drink-carton':       # horizontally
                xyz.append([x, y, 30-z, 0.7071067, 0, 0, 0.7071069])
            elif models[k] == 'chain':
                chain = random.choice(['chain', 'chain2'])
                xyz.append([x, y, 30-z, random.random(),random.random(),random.random(), random.random()])
            else:
                xyz.append([x, y, 30-z, random.random(),random.random(),random.random(), random.random()])

        spawn_model('ocean', 0, 0, 130-h)
        for k in range(samples):
            if models[k] == 'chain':
                spawn_model(chain, xyz[k][0], xyz[k][1], xyz[k][2], xyz[k][3], xyz[k][4], xyz[k][5])
            else:
                spawn_model(MODELS[CLASSES[models[k]]], xyz[k][0], xyz[k][1], xyz[k][2], xyz[k][3], xyz[k][4], xyz[k][5])
        rospy.sleep(1)  # delay

        img_msg = rospy.wait_for_message("/blueview_p900/sonar_image", Image)
        image = image_callback(img_msg)

        cv2.imwrite(save_path + 'imgs/sonar_image_' + str(i) + '.png', image)
        print(save_path + 'imgs/sonar_image_' + str(i) + '.png' + ' saved!')
        i = i + 1

        delete_model('ocean')
        rospy.sleep(1)  # delay
        if samples == 1:
            res = get_mask(CLASSES[models[0]])
            if models[0] == 'chain':
                delete_model(chain)   # delete model
            else:
                delete_model(MODELS[CLASSES[models[0]]])
        else:
            mask = get_mask(150)
            for k in range(samples):
                if models[k] == 'chain':
                    delete_model(chain)   # delete model
                else:
                    delete_model(MODELS[CLASSES[models[k]]])   # delete model
            for k in range(samples):
                if models[k] == 'chain':
                    spawn_model(chain, xyz[k][0], xyz[k][1], xyz[k][2], xyz[k][3], xyz[k][4], xyz[k][5])
                else:
                    spawn_model(MODELS[CLASSES[models[k]]], xyz[k][0], xyz[k][1], xyz[k][2], xyz[k][3], xyz[k][4], xyz[k][5])
                rospy.sleep(1)  # delay
                masks.append(get_mask(150))
                if models[k] == 'chain':
                    delete_model(chain)   # delete model
                else:
                    delete_model(MODELS[CLASSES[models[k]]])   # delete model
            res = np.zeros((480, 320), np.uint8)
            for k in range(samples):
                masks[k] = cv2.bitwise_and(mask, masks[k])
                masks[k][masks[k] > 0] = CLASSES[models[k]]
                print(models[k])
                res[(res>0) & (masks[k]>0)] = 0
                res = cv2.bitwise_or(masks[k], res)
        cv2.imwrite(save_path + 'masks/sonar_image_' + str(j) + '.png', res)
        print(save_path + 'masks/sonar_image_' + str(j) + '.png' + ' saved!')
        j = j + 1

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass