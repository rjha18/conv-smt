import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def SPN(input_fmap, theta, H, W):
    """
    Spatial Transformer Network layer implementation as described in [1].
    The layer is composed of 3 elements:
    - localization_net: takes the original image as input and outputs
        the parameters of the affine transformation that should be applied
        to the input image.
    - affine_grid_generator: generates a grid of (x,y) coordinates that
        correspond to a set of points where the input should be sampled
        to produce the transformed output.
    - bilinear_sampler: takes as input the original image and the grid
        and produces the output image using bilinear interpolation.
    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
        transformer layer is at the beginning of architecture. Should be
        a tensor of shape (B, H, W, C).
    - theta: affine transform tensor of shape (B, 6). Permits cropping,
        translation and isotropic scaling. Initialize to identity matrix.
        It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
            (https://arxiv.org/abs/1506.02025)
    """
    # grab input dimensions
    # B = tf.shape(input_fmap)[0]

    # reshape theta to (B, 2, 3)
    # theta = tf.reshape(theta, [B, 2, 3])

    # generate grids of same size or upsample/downsample if specified
    '''
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
    '''

    batch_tgrids, batch_grids, mats = affine_grid_generator(H, W, theta)

    x_s = batch_tgrids[:, 0, :, :]
    y_s = batch_tgrids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap  # , batch_tgrids, batch_grids, mats


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """

    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def generate_grid(height, width):
    x_H = np.linspace(-1.0, 1.0, height)
    x_W = np.linspace(-1.0, 1.0, width)

    axes = [x_H, x_W]
    packed_grid = np.meshgrid(*axes)

    np_H = packed_grid[0].reshape([-1, 1])
    np_W = packed_grid[1].reshape([-1, 1])

    np_grid = np.concatenate([np_H, np_W], axis=1)

    grid = tf.constant(np.float32(np_grid))

    return grid


def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    theta = tf.cast(theta, 'float32')

    base_grid = generate_grid(height, width)
    grid = tf.expand_dims(base_grid, axis=0)
    grid = tf.expand_dims(grid, axis=-1)

    b = tf.slice(theta, [0, 0], [-1, 2])
    theta_rot = tf.slice(theta, [0, 2], [-1, 1])
    theta_s = tf.slice(theta, [0, 3], [-1, 2])
    theta_z = tf.slice(theta, [0, 5], [-1, 1])

    cos_rot = tf.reshape(tf.cos(theta_rot), [-1, 1, 1])
    sin_rot = tf.reshape(tf.sin(theta_rot), [-1, 1, 1])

    A_rot1 = tf.concat([cos_rot, -sin_rot], axis=-1)
    A_rot2 = tf.concat([sin_rot, cos_rot], axis=-1)
    A_rot = tf.concat([A_rot1, A_rot2], axis=1)

    # Note the +1.0 in scale. This makes x=0 represent the identity
    # transformation
    A_s = tf.linalg.diag(theta_s+1.0)

    theta_z = tf.reshape(theta_z, [-1, 1, 1])
    A_z1 = tf.concat([tf.ones_like(theta_z), theta_z], axis=-1)
    A_z2 = tf.concat([tf.zeros_like(theta_z), tf.ones_like(theta_z)], axis=-1)
    A_z = tf.concat([A_z1, A_z2], axis=1)

    A = tf.matmul(A_z, A_rot)
    A = tf.matmul(A_s, A)

    A = tf.reshape(A, [-1, 1, 2, 2])
    b = tf.reshape(b, [-1, 1, 2, 1])

    # b_pre = tf.ones((1, 1, 2, 1))*0.5
    # b_suf = -tf.ones((1, 1, 2, 1))*0.5

    grid = tf.matmul(A, grid)+b  # -b_pre)+b#+b_suf
    grid = tf.squeeze(grid, axis=-1)
    grid = tf.transpose(grid, [0, 2, 1])

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(grid, [num_batch, 2, height, width])

    return batch_grids, base_grid, [A_rot, A_s, A_z, b]


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out
