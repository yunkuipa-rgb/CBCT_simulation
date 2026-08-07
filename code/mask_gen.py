import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import cv2

def generate_3d_metal_mask(image_shape, body_mask=None, metal_type='random', 
                          num_objects=None, metal_params=None):
    """
    Generate realistic 3D metal object masks for CBCT simulation
    
    Parameters:
    -----------
    image_shape : tuple
        Shape of the output mask (depth, height, width) for 3D
    body_mask : ndarray, optional
        Binary mask indicating body region where metals can be placed
    metal_type : str
        Type of metal objects ('dental', 'orthopedic', 'surgical', 'random', 'mixed')
    num_objects : int, optional
        Number of metal objects (auto-determined if None)
    metal_params : dict, optional
        Parameters controlling metal object properties
    
    Returns:
    --------
    metal_mask : ndarray
        3D binary mask with metal objects
    """
    
    # Default parameters
    default_params = {
        'min_radius': 2,
        'max_radius': 8,
        'min_length': 5,
        'max_length': 20,
        'intensity_variation': 0.2,
        'shape_irregularity': 0.1,
        'clustering_probability': 0.3,
    }
    
    if metal_params is None:
        metal_params = default_params
    else:
        # Merge with defaults
        for key, value in default_params.items():
            if key not in metal_params:
                metal_params[key] = value
    
    # Handle both 2D and 3D cases
    if len(image_shape) == 2:
        # Convert 2D to 3D with single slice
        image_shape = (1,) + image_shape
        is_2d_input = True
    else:
        is_2d_input = False
    
    depth, height, width = image_shape
    metal_mask = np.zeros(image_shape, dtype=np.float32)
    
    # Create default body mask if not provided
    if body_mask is None:
        body_mask = create_default_3d_body_mask(image_shape)
    elif body_mask.shape != image_shape:
        if len(body_mask.shape) == 2 and is_2d_input:
            body_mask = body_mask[np.newaxis, :, :]
        else:
            raise ValueError(f"Body mask shape {body_mask.shape} doesn't match image shape {image_shape}")
    
    # Ensure body mask has valid regions
    if np.sum(body_mask) == 0:
        print("Warning: No valid body region found for metal placement")
        return metal_mask.squeeze() if is_2d_input else metal_mask
    
    # Get valid placement coordinates
    valid_coords = np.where(body_mask > 0)
    if len(valid_coords[0]) == 0:
        print("Warning: No valid coordinates for metal placement")
        return metal_mask.squeeze() if is_2d_input else metal_mask
    
    # Determine number of metal objects
    if num_objects is None:
        num_objects = determine_3d_metal_count(metal_type, image_shape)
    
    # Generate metal objects based on type
    if metal_type == 'dental':
        metal_mask = generate_3d_dental_metals(metal_mask, valid_coords, num_objects, metal_params)
    elif metal_type == 'orthopedic':
        metal_mask = generate_3d_orthopedic_metals(metal_mask, valid_coords, num_objects, metal_params)
    elif metal_type == 'surgical':
        metal_mask = generate_3d_surgical_metals(metal_mask, valid_coords, num_objects, metal_params)
    elif metal_type == 'mixed':
        metal_mask = generate_3d_mixed_metals(metal_mask, valid_coords, num_objects, metal_params)
    else:  # random
        metal_mask = generate_3d_random_metals(metal_mask, valid_coords, num_objects, metal_params)
    
    # Apply post-processing
    metal_mask = post_process_3d_metal_mask(metal_mask, metal_params)
    
    # Return appropriate dimensions
    return metal_mask.squeeze() if is_2d_input else metal_mask

def create_default_3d_body_mask(image_shape):
    """Create a default ellipsoidal body mask"""
    depth, height, width = image_shape
    
    # Create 3D ellipsoid
    z, y, x = np.ogrid[:depth, :height, :width]
    z_center, y_center, x_center = depth // 2, height // 2, width // 2
    
    # Ellipsoid parameters (make it body-like)
    z_radius = depth * 0.4
    y_radius = height * 0.35
    x_radius = width * 0.35
    
    # Create ellipsoid
    ellipsoid = ((z - z_center) / z_radius) ** 2 + \
                ((y - y_center) / y_radius) ** 2 + \
                ((x - x_center) / x_radius) ** 2 <= 1
    
    return ellipsoid.astype(np.float32)

def determine_3d_metal_count(metal_type, image_shape):
    """Determine appropriate number of metal objects based on type and image size"""
    volume = np.prod(image_shape)
    base_density = volume / (128 ** 3)  # Normalize to 128^3 volume
    
    if metal_type == 'dental':
        return max(1, int(np.random.randint(3, 8) * base_density))
    elif metal_type == 'orthopedic':
        return max(1, int(np.random.randint(1, 4) * base_density))
    elif metal_type == 'surgical':
        return max(1, int(np.random.randint(2, 6) * base_density))
    elif metal_type == 'mixed':
        return max(1, int(np.random.randint(2, 8) * base_density))
    else:  # random
        return max(1, int(np.random.randint(1, 6) * base_density))

def generate_3d_dental_metals(metal_mask, valid_coords, num_objects, params):
    """Generate dental metal objects (fillings, crowns, implants)"""
    depth, height, width = metal_mask.shape
    
    for _ in range(num_objects):
        # Choose random valid position
        idx = np.random.randint(0, len(valid_coords[0]))
        z, y, x = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        
        # Dental metals are typically small and irregular
        metal_type = np.random.choice(['filling', 'crown', 'implant'], p=[0.5, 0.3, 0.2])
        
        if metal_type == 'filling':
            # Small, irregular filling
            radius = np.random.randint(2, 5)
            metal_mask = add_3d_irregular_sphere(metal_mask, (z, y, x), radius, 
                                               irregularity=0.3, intensity=1.0)
        
        elif metal_type == 'crown':
            # Crown-like structure
            radius = np.random.randint(3, 6)
            height_crown = np.random.randint(2, 5)
            metal_mask = add_3d_cylindrical_object(metal_mask, (z, y, x), radius, 
                                                 height_crown, intensity=1.0)
        
        else:  # implant
            # Long cylindrical implant
            radius = np.random.randint(1, 3)
            length = np.random.randint(8, 15)
            metal_mask = add_3d_cylindrical_object(metal_mask, (z, y, x), radius, 
                                                 length, intensity=1.0, vertical=True)
    
    return metal_mask

def generate_3d_orthopedic_metals(metal_mask, valid_coords, num_objects, params):
    """Generate orthopedic metal objects (screws, plates, rods)"""
    depth, height, width = metal_mask.shape
    
    for _ in range(num_objects):
        # Choose random valid position
        idx = np.random.randint(0, len(valid_coords[0]))
        z, y, x = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        
        # Orthopedic metals are typically larger and more structured
        metal_type = np.random.choice(['screw', 'plate', 'rod'], p=[0.4, 0.3, 0.3])
        
        if metal_type == 'screw':
            # Cylindrical screw
            radius = np.random.randint(1, 3)
            length = np.random.randint(10, 20)
            metal_mask = add_3d_cylindrical_object(metal_mask, (z, y, x), radius, 
                                                 length, intensity=1.0)
        
        elif metal_type == 'plate':
            # Flat plate structure
            thickness = np.random.randint(1, 3)
            plate_width = np.random.randint(8, 15)
            plate_length = np.random.randint(15, 25)
            metal_mask = add_3d_plate_object(metal_mask, (z, y, x), 
                                           (thickness, plate_width, plate_length), intensity=1.0)
        
        else:  # rod
            # Long rod (intramedullary nail)
            radius = np.random.randint(2, 4)
            length = np.random.randint(20, 40)
            metal_mask = add_3d_cylindrical_object(metal_mask, (z, y, x), radius, 
                                                 length, intensity=1.0, vertical=True)
    
    return metal_mask

def generate_3d_surgical_metals(metal_mask, valid_coords, num_objects, params):
    """Generate surgical metal objects (clips, staples, wires)"""
    depth, height, width = metal_mask.shape
    
    for _ in range(num_objects):
        # Choose random valid position
        idx = np.random.randint(0, len(valid_coords[0]))
        z, y, x = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        
        # Surgical metals are typically small and varied
        metal_type = np.random.choice(['clip', 'staple', 'wire'], p=[0.4, 0.3, 0.3])
        
        if metal_type == 'clip':
            # Small spherical clip
            radius = np.random.randint(1, 3)
            metal_mask = add_3d_sphere(metal_mask, (z, y, x), radius, intensity=1.0)
        
        elif metal_type == 'staple':
            # U-shaped staple
            thickness = 1
            width = np.random.randint(3, 6)
            length = np.random.randint(5, 10)
            metal_mask = add_3d_u_shaped_object(metal_mask, (z, y, x), 
                                              (thickness, width, length), intensity=1.0)
        
        else:  # wire
            # Thin wire structure
            radius = 1
            length = np.random.randint(8, 20)
            # Random orientation
            orientation = np.random.choice(['horizontal', 'vertical', 'diagonal'])
            metal_mask = add_3d_wire_object(metal_mask, (z, y, x), radius, length, 
                                          orientation, intensity=1.0)
    
    return metal_mask

def generate_3d_mixed_metals(metal_mask, valid_coords, num_objects, params):
    """Generate mixed types of metal objects"""
    # Divide objects among different types
    dental_count = num_objects // 3
    ortho_count = num_objects // 3
    surgical_count = num_objects - dental_count - ortho_count
    
    metal_mask = generate_3d_dental_metals(metal_mask, valid_coords, dental_count, params)
    metal_mask = generate_3d_orthopedic_metals(metal_mask, valid_coords, ortho_count, params)
    metal_mask = generate_3d_surgical_metals(metal_mask, valid_coords, surgical_count, params)
    
    return metal_mask

def generate_3d_random_metals(metal_mask, valid_coords, num_objects, params):
    """Generate random metal objects with various shapes"""
    depth, height, width = metal_mask.shape
    
    for _ in range(num_objects):
        # Choose random valid position
        idx = np.random.randint(0, len(valid_coords[0]))
        z, y, x = valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]
        
        # Random shape
        shape_type = np.random.choice(['sphere', 'cylinder', 'ellipsoid'])
        
        if shape_type == 'sphere':
            radius = np.random.randint(params['min_radius'], params['max_radius'])
            metal_mask = add_3d_sphere(metal_mask, (z, y, x), radius, intensity=1.0)
        
        elif shape_type == 'cylinder':
            radius = np.random.randint(params['min_radius'], params['max_radius'])
            length = np.random.randint(params['min_length'], params['max_length'])
            metal_mask = add_3d_cylindrical_object(metal_mask, (z, y, x), radius, 
                                                 length, intensity=1.0)
        
        else:  # ellipsoid
            radii = [np.random.randint(params['min_radius'], params['max_radius']) for _ in range(3)]
            metal_mask = add_3d_ellipsoid(metal_mask, (z, y, x), radii, intensity=1.0)
    
    return metal_mask

# 3D shape generation functions
def add_3d_sphere(mask, center, radius, intensity=1.0):
    """Add a 3D sphere to the mask"""
    z, y, x = center
    depth, height, width = mask.shape
    
    # Create coordinate grids
    zz, yy, xx = np.ogrid[:depth, :height, :width]
    
    # Sphere equation
    sphere = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    
    mask[sphere] = intensity
    return mask

def add_3d_irregular_sphere(mask, center, radius, irregularity=0.2, intensity=1.0):
    """Add an irregular 3D sphere (like dental filling)"""
    # Start with regular sphere
    mask = add_3d_sphere(mask, center, radius, intensity)
    
    # Add irregularity by random erosion/dilation
    z, y, x = center
    depth, height, width = mask.shape
    
    # Extract region around the sphere
    z_min, z_max = max(0, z - radius - 2), min(depth, z + radius + 3)
    y_min, y_max = max(0, y - radius - 2), min(height, y + radius + 3)
    x_min, x_max = max(0, x - radius - 2), min(width, x + radius + 3)
    
    region = mask[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Apply random morphological operations
    if np.random.random() < irregularity:
        kernel = morphology.ball(1)
        if np.random.random() < 0.5:
            region = morphology.erosion(region, kernel)
        else:
            region = morphology.dilation(region, kernel)
    
    mask[z_min:z_max, y_min:y_max, x_min:x_max] = region
    return mask

def add_3d_cylindrical_object(mask, center, radius, length, intensity=1.0, vertical=False):
    """Add a 3D cylindrical object"""
    z, y, x = center
    depth, height, width = mask.shape
    
    if vertical:
        # Cylinder along z-axis
        for dz in range(-length//2, length//2 + 1):
            new_z = z + dz
            if 0 <= new_z < depth:
                # Create circle at this z-level
                yy, xx = np.ogrid[:height, :width]
                circle = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
                mask[new_z, circle] = intensity
    else:
        # Cylinder along x-axis (default)
        for dx in range(-length//2, length//2 + 1):
            new_x = x + dx
            if 0 <= new_x < width:
                # Create circle in yz-plane
                zz, yy = np.ogrid[:depth, :height]
                circle = (zz - z) ** 2 + (yy - y) ** 2 <= radius ** 2
                mask[circle, new_x] = intensity
    
    return mask

def add_3d_plate_object(mask, center, dimensions, intensity=1.0):
    """Add a 3D plate-like object"""
    z, y, x = center
    thickness, plate_width, plate_length = dimensions
    depth, height, width = mask.shape
    
    # Define plate boundaries
    z_min = max(0, z - thickness//2)
    z_max = min(depth, z + thickness//2 + 1)
    y_min = max(0, y - plate_width//2)
    y_max = min(height, y + plate_width//2 + 1)
    x_min = max(0, x - plate_length//2)
    x_max = min(width, x + plate_length//2 + 1)
    
    mask[z_min:z_max, y_min:y_max, x_min:x_max] = intensity
    return mask

def add_3d_ellipsoid(mask, center, radii, intensity=1.0):
    """Add a 3D ellipsoid"""
    z, y, x = center
    rz, ry, rx = radii
    depth, height, width = mask.shape
    
    # Create coordinate grids
    zz, yy, xx = np.ogrid[:depth, :height, :width]
    
    # Ellipsoid equation
    ellipsoid = ((zz - z) / rz) ** 2 + ((yy - y) / ry) ** 2 + ((xx - x) / rx) ** 2 <= 1
    
    mask[ellipsoid] = intensity
    return mask

def add_3d_u_shaped_object(mask, center, dimensions, intensity=1.0):
    """Add a U-shaped object (like surgical staple)"""
    z, y, x = center
    thickness, width, length = dimensions
    depth, height, width_mask = mask.shape
    
    # Create the three parts of the U
    # Bottom horizontal part
    add_3d_plate_object(mask, center, (thickness, thickness, length), intensity)
    
    # Left vertical part
    left_center = (z, y - width//2, x - length//2)
    add_3d_plate_object(mask, left_center, (thickness, width, thickness), intensity)
    
    # Right vertical part
    right_center = (z, y - width//2, x + length//2)
    add_3d_plate_object(mask, right_center, (thickness, width, thickness), intensity)
    
    return mask

def add_3d_wire_object(mask, center, radius, length, orientation, intensity=1.0):
    """Add a wire-like object"""
    if orientation == 'horizontal':
        return add_3d_cylindrical_object(mask, center, radius, length, intensity, vertical=False)
    elif orientation == 'vertical':
        return add_3d_cylindrical_object(mask, center, radius, length, intensity, vertical=True)
    else:  # diagonal
        # Create diagonal wire by connecting points
        z, y, x = center
        depth, height, width = mask.shape
        
        # Define diagonal direction
        dz, dy, dx = np.random.choice([-1, 0, 1], 3)
        if dz == 0 and dy == 0 and dx == 0:
            dx = 1  # Default direction
        
        # Normalize direction
        norm = np.sqrt(dz**2 + dy**2 + dx**2)
        dz, dy, dx = dz/norm, dy/norm, dx/norm
        
        # Create wire along diagonal
        for i in range(-length//2, length//2 + 1):
            new_z = int(z + i * dz)
            new_y = int(y + i * dy)
            new_x = int(x + i * dx)
            
            if 0 <= new_z < depth and 0 <= new_y < height and 0 <= new_x < width:
                # Add small sphere at each point
                mask = add_3d_sphere(mask, (new_z, new_y, new_x), radius, intensity)
        
        return mask

def post_process_3d_metal_mask(metal_mask, params):
    """Apply post-processing to make the metal mask more realistic"""
    # Smooth the mask slightly to remove sharp edges
    if params.get('smooth_metals', True):
        sigma = params.get('smoothing_sigma', 0.5)
        metal_mask = ndimage.gaussian_filter(metal_mask, sigma=sigma)
        metal_mask = (metal_mask > 0.5).astype(np.float32)
    
    # Add intensity variations
    if params.get('intensity_variation', 0) > 0:
        variation = params['intensity_variation']
        noise = np.random.normal(1.0, variation, metal_mask.shape)
        metal_mask = metal_mask * noise
        metal_mask = np.clip(metal_mask, 0, 1)
    
    return metal_mask

# Convenience function to replace your original add_mask function
def add_3d_metal_objects(image, body_mask, metal_type='random', num_objects=None, 
                        min_objects=0, max_objects=6):
    """
    Improved replacement for your original add_mask function
    
    Parameters:
    -----------
    image : ndarray
        Input image (used for shape reference)
    body_mask : ndarray
        Binary mask indicating valid regions for metal placement
    metal_type : str
        Type of metal objects to generate
    num_objects : int, optional
        Exact number of objects (if None, randomly chosen)
    min_objects : int
        Minimum number of objects
    max_objects : int
        Maximum number of objects
    
    Returns:
    --------
    metal_mask : ndarray
        Binary mask with metal objects
    """
    
    if num_objects is None:
        num_objects = np.random.randint(min_objects, max_objects + 1)
    
    # Handle edge case
    if num_objects == 0:
        return np.zeros_like(image)
    
    # Create metal mask
    metal_mask = generate_3d_metal_mask(
        image_shape=image.shape,
        body_mask=body_mask,
        metal_type=metal_type,
        num_objects=num_objects
    )
    
    return metal_mask

# Example usage
def example_usage():
    """Example of how to use the new metal mask generator"""
    
    # Create a sample 3D volume
    image_shape = (64, 256, 256)  # (depth, height, width)
    
    # Create a body mask (ellipsoid)
    body_mask = create_default_3d_body_mask(image_shape)
    
    # Generate different types of metal masks
    dental_mask = generate_3d_metal_mask(image_shape, body_mask, 'dental', num_objects=5)
    ortho_mask = generate_3d_metal_mask(image_shape, body_mask, 'orthopedic', num_objects=3)
    mixed_mask = generate_3d_metal_mask(image_shape, body_mask, 'mixed', num_objects=8)
    
    print(f"Dental mask: {np.sum(dental_mask)} voxels")
    print(f"Orthopedic mask: {np.sum(ortho_mask)} voxels")
    print(f"Mixed mask: {np.sum(mixed_mask)} voxels")
    
    return dental_mask, ortho_mask, mixed_mask

if __name__ == "__main__":
    example_usage()