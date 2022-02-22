import numpy as np
import math
import os
import imageio
import cv2


def get_camera_external_paramter(target, origin, up):
    '''
    Transform target, origin, up to R,T
    :param target : lookat/center/target
    :param origin : eye/origin
    :param up : up
    :return : R(3*3 matrix) T(3*1 vector)
    '''
    z = np.array(origin-target, dtype=np.float32)
    z = z / np.linalg.norm(z)
    x = np.cross(z,up)
    x = x / np.linalg.norm(x)
    y = np.cross(z,x)
    y = y / np.linalg.norm(y)
    R = np.eye(3, dtype=np.float32)
    R[0:3,0:3] = [x,y,z]
    T = -target
    return R, T

def RT2affine(R, T):
    '''
    Transform R,T to affine matrix(4*4)
    :param R : camera rotation
    :param T : camera translation
    :return : affine matrix(4*4)
    '''
    matrix_r = np.eye(4, dtype=np.float32)
    matrix_r[0:3,0:3] = R
    matrix_t = np.eye(4, dtype=np.float32)
    matrix_t[0:3,3] = T
    matrix = np.dot(matrix_r,matrix_t)
    return matrix


def get_camera_external_paramter_matrix(target, origin, up):
    """
    Args:
        target (np.array): lookat/center/target
        origin (np.array): eye/origin
        up (np.array): up

    Returns:
        np.array: camera external paramter matrix
    """
    R, T = get_camera_external_paramter(target, origin, up)
    matirx = RT2affine(R, T)
    return matirx


def get_camera_internal_parameter(fov, width, height):
    """
        Args:
            fov (float): denotes the camera’s field of view in degrees, fov maps to the x-axis in screen space
            width (int/float):
            height (int/float):

        Returns:
            fx, fy, ux, uy
    """
    fov_ = fov * math.pi / 180
    focal = (width / 2) / math.tan(fov_ / 2)

    fx = focal
    fy = focal
    ux = width / 2
    uy = height / 2

    return fx, fy, ux, uy


def get_camera_internal_parameter_matrix(fx, fy, ux, uy):
    """
    Args:
        fx:
        fy:
        ux:
        uy:

    Returns:
        np.array: camera internal parameter matrix
    """
    matrix = np.array([
        [fx,  0., ux],
        [ 0., fy, uy],
        [ 0.,  0.,  1]
    ], dtype=np.float32)

    return matrix




if __name__ == '__main__':
    r_theta_x = 0
    r_theta_y = np.pi/4
    r_theta_z = 0
    center = np.array([0., 0., 500.])

    c, s = np.cos(r_theta_x), np.sin(r_theta_x)
    Rx = np.array([[1, 0, 0, 0],
                   [0, c, s, 0],
                   [0, -s, c, 0],
                   [0, 0, 0, 1]])

    c, s = np.cos(r_theta_y), np.sin(r_theta_y)
    Ry = np.array([[c, 0, -s, 0],
                   [0, 1, 0, 0],
                   [s, 0, c, 0],
                   [0, 0, 0, 1]])

    c, s = np.cos(r_theta_z), np.sin(r_theta_z)
    Rz = np.array([[c, s, 0, 0],
                   [-s, c, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0,1]])

    to_center = np.array([[1, 0, 0, -center[0]],
                          [0, 1, 0, -center[1]],
                          [0, 0, 1, -center[2]],
                          [0, 0, 0, 1]])

    transform_mat = np.linalg.inv(to_center)@Rz@Ry@Rx@to_center
    # transform_mat = np.array(transform_mat, dtype=np.int)
    print(transform_mat)

    target1 = np.array([-500, 0, 0])
    origin = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    matrix1 = get_camera_external_paramter_matrix(target1, origin, up)
    target2 = np.array([-353.55339, 0, -353.55339])
    matrix2 = get_camera_external_paramter_matrix(target2, origin, up)
    matrix = np.matmul(matrix2, np.linalg.inv(matrix1))
    print(matrix)

    matrix = np.matmul(matrix2, np.linalg.inv(matrix1))


    target1 = np.array([-501, 0, 0])
    origin = np.array([-1, 0, 0])
    up = np.array([0, 1, 0])
    matrix1 = get_camera_external_paramter_matrix(target1, origin, up)
    # print(matrix1)

    target2 = np.array([-1, 0, 500])
    matrix2 = get_camera_external_paramter_matrix(target2, origin, up)
    # print(matrix2)
    matrix = np.matmul(matrix2, np.linalg.inv(matrix1))
    # print(matrix)

    target1 = np.array([-500, 0, 0])
    origin = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    matrix1 = get_camera_external_paramter_matrix(target1, origin, up)
    # print(matrix1)
    coor1 = np.matmul(matrix1, np.array([1, 1, 1, 1]))
    ans1 = np.array([1., -1., 501., 1.])
    assert (np.abs(coor1-ans1)<1e-8).all()


    target2 = np.array([0, 0, 500])
    matrix2 = get_camera_external_paramter_matrix(target2, origin, up)
    coor2 = np.matmul(matrix2, np.array([1, 1, 1, 1]))
    ans2 = np.array([1., -1., 499., 1.])
    assert (np.abs(coor2-ans2)<1e-8).all()
    matrix = np.matmul(matrix2, np.linalg.inv(matrix1))
    # print(matrix)

    matrix = np.matmul(matrix2, np.linalg.inv(matrix1))
    # print(matrix)
    assert (np.abs(coor2-np.matmul(matrix, coor1))<1e-8).all()

    target1 = np.array([-500, 0, 0])
    origin = np.array([0, 0, 0])
    up = np.array([0, -1, 0])
    matrix1 = get_camera_external_paramter_matrix(target1, origin, up)
    coor1 = np.matmul(matrix1, np.array([1, 1, 1, 1]))
    ans1 = np.array([-1., 1., 501., 1.])
    assert (np.abs(coor1-ans1)<1e-8).all()

    target2 = np.array([0, 0, 500])
    matrix2 = get_camera_external_paramter_matrix(target2, origin, up)
    coor2 = np.matmul(matrix2, np.array([1, 1, 1, 1]))
    ans2 = np.array([-1., 1., 499., 1.])
    assert (np.abs(coor2-ans2)<1e-8).all()

    target1 = np.array([-501, 0, 0])
    origin = np.array([-1, 0, 0])
    up = np.array([0, 1, 1])
    matrix1 = get_camera_external_paramter_matrix(target1, origin, up)
    coor1 = np.matmul(matrix1, np.array([1, 1, 1, 1]))
    ans1 = np.array([0., -1.41421356, 502., 1.])
    assert (np.abs(coor1-ans1)<1e-8).all()

    target1 = np.array([-501, 1, 1])
    origin = np.array([-1, 1, 1])
    up = np.array([0, 1, 0])
    matrix1 = get_camera_external_paramter_matrix(target1, origin, up)
    coor = np.array([1, 1, 1, 1])
    coor1 = np.matmul(matrix1, coor)
    ans1 = np.array([0., 0., 502., 1.])
    assert ((coor1-ans1)<1e-8).all()


    target2 = np.array([0, 0, 499])
    matrix2 = get_camera_external_paramter_matrix(target2, origin, up)


    fov = 30
    width = 128
    height = 128

    internal_matrix = np.eye(4)
    fx, fy, ux, uy = get_camera_internal_parameter(fov, width, height)
    print(fx, fy, ux, uy)
    tmp = get_camera_internal_parameter_matrix(fx, fy, ux, uy)
    internal_matrix[:3,:3] = tmp

    # print(internal_matrix)

    dataset_dir = '/home/acc/cj/MultiviewRender/render_result'
    views = np.load(os.path.join(dataset_dir, '../views.npy'), allow_pickle=True, encoding='latin1').item()
    for (key, value) in views.items():
        if(key!='reg_deng/RGB/hand_1/sample5'):
            continue
        print(key)
        path = os.path.join(dataset_dir, key)
        print(path)
        img_view0 = cv2.imread(path+'_view0.exr', cv2.IMREAD_ANYDEPTH)
        img_view1 = cv2.imread(path+'_view1.exr', cv2.IMREAD_ANYDEPTH)
        print(value['origin'])
        print(value['view0'])
        print(value['view1'])
        print(value['up'])
        external_matrix0 = get_camera_external_paramter_matrix(value['view0'], value['origin'], value['up'])
        external_matrix1 = get_camera_external_paramter_matrix(value['view1'], value['origin'], value['up'])
        internal_matrix = np.eye(4)
        tmp = get_camera_internal_parameter_matrix(fx, fy, ux, uy)
        internal_matrix[:3,:3] = tmp
        
        matrix0 = np.matmul(internal_matrix, external_matrix0)
        matrix1 = np.matmul(internal_matrix, external_matrix1)
        matrix = np.matmul(matrix1, np.linalg.inv(matrix0))

        print(img_view0[64,64])
        u, v = 57, 87
        print(img_view0[v, u])
        zc = img_view0[v, u]
        point0 = np.array([u*zc, v*zc, zc, 1])
        pointc0 = np.matmul(np.linalg.inv(internal_matrix), point0)
        print(pointc0)
        point = np.matmul(np.linalg.inv(external_matrix0), pointc0)
        print(point)
        pointc1 = np.matmul(external_matrix1, point)
        print(pointc1)
        point1 = np.matmul(internal_matrix, pointc1)
        point1[:2] = point1[:2]/point1[2]
        print(point1)
        u, v = int(point1[0]), int(point1[1])
        # print(img_view1[v-3:v+3, u-3:u+3])
        print(img_view1[v, u])

        break
