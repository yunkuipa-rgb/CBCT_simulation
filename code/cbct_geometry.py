import numpy as np
import odl
from odl.contrib import torch as odl_torch


class initialization:
    def __init__(self, n_proj):
        self.param = {}
        self.param['reso_x'] = 1
        self.param['reso_y'] = 1
        self.param['reso_z'] = 2

        # image dimensions
        self.param['nx_h'] = 512
        self.param['ny_h'] = 512
        
        # Set nz_h based on geometry type
        self.param['nz_h'] = 256
            
        # Physical dimensions
        self.param['sx'] = self.param['nx_h'] * self.param['reso_x']
        self.param['sy'] = self.param['ny_h'] * self.param['reso_y']
        self.param['sz'] = self.param['nz_h'] * self.param['reso_z']

        # view parameters
        self.param['startangle'] = 0
        self.param['endangle'] = 2 * np.pi
        self.param['nProj'] = n_proj if n_proj else 64

        # detector parameters
        self.param['margin'] = 1.2
        self.param['nu_h'] = 512
        self.param['dde'] = 600  # detector-to-object distance
        self.param['dso'] = 600  # source-to-object distance
        object_diagonal = np.sqrt(self.param['sx']**2 + self.param['sy']**2)
        magnification = (self.param['dso'] + self.param['dde']) / self.param['dso']
        self.param['su'] = object_diagonal * magnification * self.param['margin']  # 20% margin


def imaging_geo_cone(param):
    """
    Similar to imaging_geo_cone but returns PyTorch-compatible operator
    """

    # Create the reconstruction space (3D volume)
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0, -param.param['sz'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0, param.param['sz'] / 2.0],
        shape=[param.param['nx_h'], param.param['ny_h'], param.param['nz_h']],
        dtype='float32')
    
    # Create angle partition
    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])
    
    # Create 2D detector partition
    detector_partition_h = odl.uniform_partition(
        min_pt=[-param.param['su'] / 2.0, -param.param['su'] / 2.0],
        max_pt=[param.param['su'] / 2.0, param.param['su'] / 2.0],
        shape=[param.param['nu_h'], param.param['nu_h']])
    
    # Create cone beam geometry
    geometry_h = odl.tomo.ConeBeamGeometry(
        angle_partition, 
        detector_partition_h,
        src_radius=param.param['dso'],
        det_radius=param.param['dde'])
    
    # Create ray transform
    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
    fbp_oper_hh = odl.tomo.fbp_op(ray_trafo_hh)

    return ray_trafo_hh, fbp_oper_hh
