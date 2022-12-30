#!/usr/bin/env python3
import rospy
import rospkg
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np
import cv2
import random
import math
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

'''
CLASSES = {"background": 0, "bottle": 1, "can": 2, "chain": 3, "drink-carton": 4,
            "hook": 5, "propeller": 6, "shampoo-bottle": 7, "standing-bottle": 8,
            "tire": 9, "valve": 10, "wall": 11}
MODELS = ['', 'bottle', 'Coke', 'chain', 'drink_carton',
          'hook', 'propeller', 'shampoo_bottle', 'standing_bottle',
          'car_wheel', 'valve', 'wall']
'''
CLASSES = {"background": 0, "bottle": 1, "can": 2, "drink-carton": 3,
            "hook": 4, "propeller": 5, "tire": 6, "valve": 7, "wall": 8}

MODELS = ['', 'bottle', 'Coke', 'drink_carton',
          'hook', 'propeller', 'car_wheel', 'valve', 'wall']

save_path = rospkg.RosPack().get_path('sonar_image_save')+'/sonar_imgs/'
th = 10
pitch = 15.0
bridge = CvBridge() # Initialize the CvBridge class
h = 0.5
max_sample = 3

# intrinsic parameter
img_w = 320
img_h = 480
f = 500
cx = float(img_w/2.0)
cy = float(img_h/2.0)

K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0,  1]])

crop_mask = np.zeros((480, 320), dtype=np.uint8)
for hh in range(480):
    for ww in range(320):
        if abs(math.atan2(ww-160, 570-hh)) <= math.radians(15) and 110 < math.dist([hh, ww],[570, 160]) < 540:
            crop_mask[hh][ww] = 1

def point_callback(msg, is_img=True):
    # log some info about the pointcloud topic
    rospy.loginfo(msg.header)

    try:
        pointcloud = []
        for data in pc2.read_points(msg, skip_nans=True):
        # x, y, z, intensity, ring, timestamp
            pointcloud.append(np.array(data[:3]))
        image = np.zeros((480, 320), dtype=np.uint8)
        pc_fat = np.transpose(pointcloud)

        rvec = np.array([(90-pitch)*np.pi/180.0, 0.0, 0.0])
        tvec = np.array([0.0, 1.8, 1.8+h])

        rmat, _ = cv2.Rodrigues(rvec)
        try:
            pc2_fat = np.matmul(rmat, pc_fat)
            pc2_fat += np.tile(np.transpose(tvec)[:,np.newaxis], (1, pc2_fat.shape[1]))
        except:
            pc2_fat = pc_fat
        
        image = get_depth_map(K, pc2_fat, image)
        image = cv2.GaussianBlur(image, (1,5), 3) # cv2.blur(image,(1,5))
        if is_img:      # Add noise
            for hh in range(480):
                for ww in range(320):
                    if image[hh][ww]:
                        image[hh][ww] = np.clip(image[hh][ww] + np.random.normal(0,10), 0, 255)
        image = image * crop_mask
        return image
    except CvBridgeError:
          rospy.logerr("CvBridge Error: {0}".format(e))

def point_npy_callback(msg):
    # log some info about the pointcloud topic
    rospy.loginfo(msg.header)
    try:
        pointcloud = []
        for data in pc2.read_points(msg, skip_nans=True):
        # x, y, z, intensity, ring, timestamp
            pointcloud.append(np.array(data[:3]))
        pointcloud = np.array(pointcloud)
        np.save('pointcloud'+str(pitch), pointcloud)
        print(pointcloud.shape)
        return pointcloud
    except CvBridgeError:
          rospy.logerr("CvBridge Error: {0}".format(e))

def get_depth_map(intrinsic, pc, img_depth):
    img_depth *= 0
    try:
        uvw = np.matmul(intrinsic, pc)
        uvw[0, :] = uvw[0, :] / uvw[2, :]
        uvw[1, :] = uvw[1, :] / uvw[2, :]
        # print(np.max(uvw[2, :])-np.min(uvw[2, :]))
        # uvw[2, :] = 255 - (uvw[2, :] - np.min(uvw[2, :])) / (np.max(uvw[2, :]) - np.min(uvw[2, :])) * 200
        uvw[2, :] = np.clip(200 - (uvw[2, :] - 2) / 1.4 * 200, 0, 150)

        uvw_t = np.transpose(uvw).astype(int)
        h, w = img_depth.shape
        mask = np.where((uvw_t[:, 0] < 0) | (uvw_t[:, 0] >= w) | (uvw_t[:, 1] < 0) | (uvw_t[:, 1] >= h))
        uvw_t = np.delete(uvw_t, mask, axis=0)
        for u in uvw_t:
            if img_depth[u[1], u[0]] < u[2]:
                img_depth[u[1], u[0]] = u[2]
        #img_depth[uvw_t[:, 1], uvw_t[:, 0]] = uvw_t[:, 2]
    except:
        pass
    return img_depth

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
    image_pub = rospy.Publisher("image_topic", Image, queue_size=5)

    q = quaternion_from_euler(0, np.deg2rad(pitch), 0)
    spawn_model('depth_camera', 0, 0, 30, q[0], q[1], q[2], q[3]) # pitch 15 degree
    rospy.sleep(1.)  # delay

    i = 0
    j = 0
    # Multi
    while True:
        samples = random.randint(1,max_sample)
        models = random.sample(['bottle', 'can', 'drink-carton', 'hook', 'propeller', 'tire', 'valve', 'wall'], samples)

        masks = []
        xyz = []

        rand_x = sorted(random.sample(range(1,max_sample+1), samples))
        for k in range(samples):
            #z = (h-0.1)-(random.random()*0.01)  # z가 h보다 0.1 작다 h는 z보다 0.1 크다
            #x = h/(((np.tan(np.deg2rad(pitch+5))-np.tan(np.deg2rad(pitch-7.5)))/h)*(rand_x[k]/max_sample)+np.tan(np.deg2rad(pitch-5))/h) + random.random()*0.1
            x = (random.random()/rand_x[k]*2+1+random.random()*0.1) # 1~2
            y = x*(np.sin(np.deg2rad(14))*(2*random.random()-1))   # -0.24~0.24

            if models[k] == 'wall':
                #xyz.append([h/(np.tan(np.deg2rad(pitch-10))) + random.random()*0.1, 5*(random.random()-0.5), 30-h+2, 0, 0, 0, 0])
                xyz.append([3.5+random.random()*0.1, 5*(random.random()-0.5), 30-h+2, 0, 0, 0, 0])
            elif models[k] == 'valve':
                xyz.append([x, y, 30-h+0.24, 0, 0, 0, 0])
            elif models[k] == 'standing-bottle':    # standing
                xyz.append([x, y, 30-h+0.1, 0,0,0,0])
            elif models[k] == 'bottle':             # lying horizontally
                xyz.append([x, y, 30-h+0.1, 0, 0, 0, 0])
            elif models[k] == 'shampoo-bottle':     # vertical
                xyz.append([x, y, 30-h+0.3, 0.7071067, 0, 0, 0.7071069])
            elif models[k] == 'tire':               # horizontally
                xyz.append([x, y, 30-h+0.05, 0, 0, 0, 0])
            elif models[k] == 'drink-carton':       # horizontally
                xyz.append([x, y, 30-h+0.2,0,0,0,0])
            elif models[k] == 'chain':
                chain = random.choice(['chain', 'chain2'])
                xyz.append([x, y, 30-h+0.1, random.random(),random.random(),random.random(), random.random()])
            else:
                xyz.append([x, y, 30-h+0.1, random.random(),random.random(),random.random(), random.random()])

        spawn_model('ocean', 0, 0, 130-h)

        for k in range(samples):
            if models[k] == 'chain':
                spawn_model(chain, xyz[k][0], xyz[k][1], xyz[k][2], xyz[k][3], xyz[k][4], xyz[k][5])
            else:
                spawn_model(MODELS[CLASSES[models[k]]], xyz[k][0], xyz[k][1], xyz[k][2], xyz[k][3], xyz[k][4], xyz[k][5])
        rospy.sleep(1)  # delay

        point_msg = rospy.wait_for_message("/camera/depth/points", PointCloud2)
        image = point_callback(point_msg)
        image_message = bridge.cv2_to_imgmsg(image, "passthrough")
        image_pub.publish(image_message)

        cv2.imwrite(save_path + 'imgs/pitch_'+str(pitch) +'_'+ str(i).zfill(4) + '.png', image)
        print(save_path + 'imgs/pitch_'+str(pitch) +'_'+str(i).zfill(4) + '.png' + ' saved!')
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
                rospy.sleep(0.8)  # delay
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
        cv2.imwrite(save_path + 'masks/pitch_'+str(pitch) +'_'+ str(j).zfill(4) + '.png', res)
        print(save_path + 'masks/pitch_'+str(pitch) +'_'+ str(j).zfill(4) + '.png' + ' saved!')
        j = j + 1
    '''
    spawn_model('ocean', 0, 0, 130-h)
    spawn_model('car_wheel', 1.65, 0, 30-h+0.05, 0, 0, 0, 0)
    rospy.sleep(1.)  # delay
    point_msg = rospy.wait_for_message("/camera/depth/points", PointCloud2)
    #image = point_npy_callback(point_msg)
    image = point_callback(point_msg)
    cv2.imwrite(save_path + 'imgs/i.png', image)
    '''

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass