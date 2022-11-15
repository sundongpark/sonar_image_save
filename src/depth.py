#!/usr/bin/env python3
import rospy
import rospkg
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
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
th = 10
pitch = 30.0
bridge = CvBridge() # Initialize the CvBridge class
h = 1.2
max_sample = 5
sonar_min_distance = 0.7
sonar_max_distance = 7.5


def point_callback(msg):
    # log some info about the pointcloud topic
    rospy.loginfo(msg.header)
    try:
        pointcloud = []
        for data in pc2.read_points(msg, skip_nans=True):
        # x, y, z, intensity, ring, timestamp
            pointcloud.append(data[:3])
        pointcloud = np.array(pointcloud)
        image = np.zeros((480, 320), dtype=np.uint8)

        for p in pointcloud:
            x = int((p[0]/(sonar_max_distance*np.sin(np.deg2rad(30/2))*2))*320+160)
            projection_y = (p[1]*np.cos( np.deg2rad(pitch)-np.arctan(p[1]/p[2]) ))
            y = int(projection_y/(h/np.cos(np.deg2rad(pitch-15/2.0)))*480+240)
            if 0 <= y < 480 and 0 <= x < 320:
                image[y][x] += 10

        image = cv2.flip(image, 1)
        image = cv2.blur(image, (1,3))
        return image
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
    point_msg = rospy.wait_for_message("/camera/depth/points", PointCloud2)
    mask = point_callback(point_msg)
    ret, mask = cv2.threshold(mask, th, class_id, cv2.THRESH_BINARY)
    return mask

def main():
    # Initialize the ROS Node named 'sonar_image_save', allow multiple nodes to be run with this name
    rospy.init_node('sonar_image_save', anonymous=True)
    
    q = quaternion_from_euler(0, np.deg2rad(pitch), 0)
    spawn_model('depth_camera', 0, 0, 30, q[0], q[1], q[2], q[3]) # pitch 15 degree
    rospy.sleep(1.)  # delay

    i = 0
    j = 0
    # Multi
    while True:
        samples = random.randint(1,max_sample)
        models = random.sample(['bottle', 'can', 'chain', 'drink-carton', 'hook', 'propeller', 'shampoo-bottle', 'standing-bottle', 'tire', 'valve', 'wall'], samples)  # except wall
        masks = []
        xyz = []

        rand_x = sorted(random.sample(range(1,max_sample+1), samples))
        for k in range(samples):
            z = (h-0.1)-(random.random()*0.01)
            x = z/(((np.tan(np.deg2rad(pitch+5))-np.tan(np.deg2rad(pitch-5)))/z)*(rand_x[k]/max_sample)+np.tan(np.deg2rad(pitch-5))/z) + random.random()*0.1

            #x = z/(0.23*(rand_x[k]/5.0)+0.15) + random.random()*0.1       #0.15~0.38 (h=0.7, 1.842~4.667)
            #print(x)
            y = x*(np.sin(np.deg2rad(14))*(2*random.random()-1))   # -0.24~0.24

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

        point_msg = rospy.wait_for_message("/camera/depth/points", PointCloud2)
        image = point_callback(point_msg)

        cv2.imwrite(save_path + 'imgs/pitch_'+str(pitch) +'_'+ str(i) + '.png', image)
        print(save_path + 'imgs/pitch_'+str(pitch) +'_'+str(i) + '.png' + ' saved!')
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
        cv2.imwrite(save_path + 'masks/pitch_'+str(pitch) +'_'+ str(j) + '.png', res)
        print(save_path + 'masks/pitch_'+str(pitch) +'_'+ str(j) + '.png' + ' saved!')
        j = j + 1

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass