import numpy as np
import cv2
from scipy.interpolate import griddata

def fisheye_to_erp(fisheye_image, intrinsics, output_size=(1024, 512)):
    # Read intrinsics
    xi = intrinsics['mirror_parameters']['xi']
    k1 = intrinsics['distortion_parameters']['k1']
    k2 = intrinsics['distortion_parameters']['k2']
    p1 = intrinsics['distortion_parameters']['p1']
    p2 = intrinsics['distortion_parameters']['p2']
    gamma1 = intrinsics['projection_parameters']['gamma1']
    gamma2 = intrinsics['projection_parameters']['gamma2']
    u0 = intrinsics['projection_parameters']['u0']
    v0 = intrinsics['projection_parameters']['v0']
            
    erp_height = output_size[1]
    erp_width = output_size[0]

    latitude = np.linspace(-np.pi / 2, np.pi / 2, erp_height)
    longitude = np.linspace(np.pi, -np.pi, erp_width)
    longitude, latitude = np.meshgrid(longitude, latitude)
    longitude = longitude.astype(np.float32)
    latitude = latitude.astype(np.float32)

    x = np.cos(latitude) * np.cos(longitude)
    z = np.cos(latitude) * np.sin(longitude)
    y = np.sin(latitude)
    
    mask = z < 0
    x = np.where(mask, 1/np.sqrt(3), x)
    y = np.where(mask, 1/np.sqrt(3), y)
    z = np.where(mask, 1/np.sqrt(3), z)

    p_u = x / (z + xi)
    p_v = y / (z + xi)

    # apply distortion
    ro2 = p_u*p_u + p_v*p_v

    p_u *= 1 + k1*ro2 + k2*ro2*ro2
    p_v *= 1 + k1*ro2 + k2*ro2*ro2

    p_u += 2*p1*p_u*p_v + p2*(ro2 + 2*p_u*p_u)
    p_v += p1*(ro2 + 2*p_v*p_v) + 2*p2*p_u*p_v

    # apply projection
    p_u = gamma1*p_u + u0 
    p_v = gamma2*p_v + v0

    # Remap the fisheye image to ERP projection
    erp_img = cv2.remap(fisheye_image, p_u, p_v, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT )

    return erp_img


def erp_to_fisheye(erp_img, intrinsics, rotate):

    width, height = intrinsics['image_width'], intrinsics['image_height']
    # erp_img =  np.roll(erp_img, 100, axis=0)

    xi = intrinsics['mirror_parameters']['xi']
    k1 = intrinsics['distortion_parameters']['k1']
    k2 = intrinsics['distortion_parameters']['k2']
    p1 = intrinsics['distortion_parameters']['p1']
    p2 = intrinsics['distortion_parameters']['p2']
    gamma1 = intrinsics['projection_parameters']['gamma1']
    gamma2 = intrinsics['projection_parameters']['gamma2']
    u0 = intrinsics['projection_parameters']['u0']
    v0 = intrinsics['projection_parameters']['v0']

    erp_width, erp_height = erp_img.shape[1], erp_img.shape[0]

    latitude = np.linspace(-np.pi / 2, np.pi / 2, erp_height)
    longitude = np.linspace(np.pi, -np.pi, erp_width)
    
    longitude, latitude = np.meshgrid(longitude, latitude)
    longitude = longitude.astype(np.float32)
    latitude = latitude.astype(np.float32)

    x = np.cos(latitude) * np.cos(longitude)
    z = np.cos(latitude) * np.sin(longitude)
    y = np.sin(latitude)  

    xyz = np.stack((x,y,z),axis=2)

    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    [R1, _] = cv2.Rodrigues(x_axis * np.radians(rotate[0]))
    [R2, _] = cv2.Rodrigues(y_axis * np.radians(rotate[1]))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape([erp_height * erp_width, 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([erp_height , erp_width, 3])

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    # remove points with z < 0 without changing the shape of the arrays
    mask_z = z < 0
    erp_masked = np.where(mask_z[..., np.newaxis], 0, erp_img)

    x = np.where(mask_z, 1/np.sqrt(3), x)
    y = np.where(mask_z, 1/np.sqrt(3), y)
    z = np.where(mask_z, 1/np.sqrt(3), z)

    # convert to axuiliary coordinates
    p_u = x / (z + xi)
    p_v = y / (z + xi)

    # apply distortion
    ro2 = p_u*p_u + p_v*p_v

    p_u *= 1 + k1*ro2 + k2*ro2*ro2
    p_v *= 1 + k1*ro2 + k2*ro2*ro2

    p_u += 2*p1*p_u*p_v + p2*(ro2 + 2*p_u*p_u)
    p_v += p1*(ro2 + 2*p_v*p_v) + 2*p2*p_u*p_v

    # apply projection
    p_u = gamma1*p_u + u0 
    p_v = gamma2*p_v + v0

    mask_dimention = (p_u < 0) | (p_u >= width) | (p_v < 0) | (p_v >= height)
    erp_masked = np.where(mask_dimention[..., np.newaxis], 0, erp_masked)

    circle_mask = np.sqrt((p_u - u0)**2 + (p_v - v0)**2) > (min(height, width) / 2 - 0.025 * min(height, width))
    erp_masked = np.where(circle_mask[..., np.newaxis], 0, erp_masked)
    
    coordinates = np.stack([p_u, p_v],axis=-1)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    coordinate_points = coordinates.reshape((-1, 2))

    # also interpolate, just some part the projection can be too sparse compared to the inverse way
    interpolated_colors = griddata(coordinate_points, erp_masked.reshape((-1, 3)), grid_points, method='linear')
    interpolated_colors = interpolated_colors.reshape((height, width, 3))

    return interpolated_colors, erp_masked, np.logical_or(np.logical_or(mask_z, mask_dimention), circle_mask)


if __name__ == '__main__':
    import glob
    import yaml

    # Read YAML file
    with open('./config/kitti360_leftfisheye.yaml', 'r') as file:
        intrinsics = yaml.safe_load(file)

    # Print the dictionary
    print(intrinsics)

    K = [[intrinsics['projection_parameters']['gamma1'], 0., intrinsics['projection_parameters']['u0']],
         [0., intrinsics['projection_parameters']['gamma2'], intrinsics['projection_parameters']['v0']],
         [0., 0., 1.]]
    distCoeffs = [intrinsics['distortion_parameters']['k1'], intrinsics['distortion_parameters']['k2'],
                  intrinsics['distortion_parameters']['p1'], intrinsics['distortion_parameters']['p2']]
    K = np.array(K).astype(np.float32)
    distCoeffs = np.array(distCoeffs).astype(np.float32)
    xi = intrinsics['mirror_parameters']['xi']


    image_files = glob.glob('./samples/kitti360left*.png')
    for image_path in image_files:
        # get erp image
        fisheye_image = cv2.imread(image_path)
        erp_img = fisheye_to_erp(fisheye_image, intrinsics, output_size=(1024, 512))
        cv2.imshow('erp', erp_img)

        # get undistorted image
        h, w = fisheye_image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 1, (w, h))

        K_new = np.array([[w / 8, 0, w / 2],
                          [0, h / 8, h / 2],
                          [0, 0, 1]]).astype(np.float32)
        undistort_img = cv2.omnidir.undistortImage(fisheye_image, K, distCoeffs, np.array(xi).astype(np.float32), cv2.omnidir.RECTIFY_PERSPECTIVE, Knew=K_new)

        # K_new = np.array([[w / 3.1415, 0, 0],
        #                   [0, h / 3.1415, 0],
        #                   [0, 0, 1]]).astype(np.float32)
        # undistort_img = cv2.omnidir.undistortImage(fisheye_image, K, distCoeffs, np.array(xi).astype(np.float32), cv2.omnidir.RECTIFY_CYLINDRICAL, Knew=K_new)
        cv2.imshow('rect', undistort_img)
        cv2.waitKey()

