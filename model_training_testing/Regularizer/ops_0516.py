import numpy as np
import tensorflow as tf

def ddx(inpt, channel, dx):
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddx1D = tf.constant([-1./2., 0., 1./2.], dtype=tf.float32)
    ddx3D = tf.reshape(ddx1D, shape=(-1,1,1,1,1))

    strides = [1,1,1,1,1]
    # var_pad = periodic_padding( var, ((3,3),(0,0),(0,0)) )
    output = tf.nn.conv3d(var, ddx3D, strides, padding = 'VALID', data_format = 'NDHWC')
    output = tf.scalar_mul(1./dx, output)
    output = output[:, :, 1:-1, 1:-1, :]
    return output

def ddy(inpt, channel, dy):
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddy1D = tf.constant([-1./2., 0., 1./2.], dtype=tf.float32)
    ddy3D = tf.reshape(ddy1D, shape=(1,-1,1,1,1))

    strides = [1,1,1,1,1]
    # var_pad = periodic_padding( var, ((0,0),(3,3),(0,0)) )
    output = tf.nn.conv3d(var, ddy3D, strides, padding = 'VALID', data_format = 'NDHWC')
    output = tf.scalar_mul(1./dy, output)
    output = output[:, 1:-1, :, 1:-1, :]
    return output

def ddz(inpt, channel, dz):
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddz1D = tf.constant([-1./2., 0., 1./2.], dtype=tf.float32)
    ddz3D = tf.reshape(ddz1D, shape=(1,1,-1,1,1))

    strides = [1,1,1,1,1]
    # var_pad = periodic_padding( var, ((0,0),(0,0),(3,3)) )
    output = tf.nn.conv3d(var, ddz3D, strides, padding = 'VALID', data_format = 'NDHWC')
    output = tf.scalar_mul(1./dz, output)
    output = output[:, 1:-1, 1:-1, :, :]
    return output

def d2dx2(inpt, channel, dx):
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddx1D = tf.constant([1., -2., 1.], dtype=tf.float32)
    ddx3D = tf.reshape(ddx1D, shape=(-1,1,1,1,1))

    strides = [1,1,1,1,1]
    # var_pad = periodic_padding( var, ((3,3),(0,0),(0,0)) )
    output = tf.nn.conv3d(var, ddx3D, strides, padding = 'VALID', data_format = 'NDHWC')
    output = tf.scalar_mul(1./dx**2, output)
    output = output[:, :, 1:-1, 1:-1, :]
    return output

def d2dy2(inpt, channel, dy):
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddy1D = tf.constant([1., -2., 1.], dtype=tf.float32)
    ddy3D = tf.reshape(ddy1D, shape=(1,-1,1,1,1))

    strides = [1,1,1,1,1]
    # var_pad = periodic_padding( var, ((0,0),(3,3),(0,0)) )
    output = tf.nn.conv3d(var, ddy3D, strides, padding = 'VALID', data_format = 'NDHWC')
    output = tf.scalar_mul(1./dy**2, output)
    output = output[:, 1:-1, :, 1:-1, :]
    return output

def d2dz2(inpt, channel, dz):
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddz1D = tf.constant([1., -2., 1.], dtype=tf.float32)
    ddz3D = tf.reshape(ddz1D, shape=(1,1,-1,1,1))

    strides = [1,1,1,1,1]
    # var_pad = periodic_padding( var, ((0,0),(0,0),(3,3)) )
    output = tf.nn.conv3d(var, ddz3D, strides, padding = 'VALID', data_format = 'NDHWC')
    output = tf.scalar_mul(1./dz**2, output)
    output = output[:, 1:-1, 1:-1, :, :]
    return output

def get_velocity_grad(inpt, dx, dy, dz):
    dudx = ddx(inpt, 0, dx)
    dudy = ddy(inpt, 0, dy)
    dudz = ddz(inpt, 0, dz)

    dvdx = ddx(inpt, 1, dx)
    dvdy = ddy(inpt, 1, dy)
    dvdz = ddz(inpt, 1, dz)

    dwdx = ddx(inpt, 2, dx)
    dwdy = ddy(inpt, 2, dy)
    dwdz = ddz(inpt, 2, dz)

    return dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz

def get_strain_rate_mag2(vel_grad):
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad

    strain_rate_mag2 = dudx**2 + dvdy**2 + dwdz**2 \
                     + 2*( (0.5*(dudy + dvdx))**2 + (0.5*(dudz + dwdx))**2 + (0.5*(dvdz + dwdy))**2 )

    return strain_rate_mag2

def get_vorticity(vel_grad):
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad
    vort_x = dwdy - dvdz
    vort_y = dudz - dwdx
    vort_z = dvdx - dudy
    return vort_x, vort_y, vort_z

def get_enstrophy(vorticity):
    omega_x, omega_y, omega_z = vorticity
    Omega = omega_x**2 + omega_y**2 + omega_z**2
    return Omega

def get_continuity_residual(vel_grad):
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad
    res = dudx + dvdy + dwdz
    return res

def get_pressure_residual(inpt, vel_grad, dx, dy, dz):
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad
    d2pdx2 =d2dx2(inpt, 3, dx)
    d2pdy2 =d2dy2(inpt, 3, dy)
    d2pdz2 =d2dz2(inpt, 3, dz)
    res = (d2pdx2 + d2pdy2 + d2pdz2)
    res = res + dudx*dudx + dvdy*dvdy + dwdz*dwdz + 2*(dudy*dvdx + dudz*dwdx + dvdz*dwdy)
    return res