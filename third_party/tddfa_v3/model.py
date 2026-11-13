"""Morphable-model decoder adapted from 3DDFA-V3 model/recon.py.

Upstream commit: e15385837dc1e051a6cf376b3827f2e279537b29.
Renderer/network setup is separate; see README.md for adaptation notes.
"""
import numpy as np
import torch
import torch.nn.functional as F


class MorphableFaceModel:
    """Decode coefficients with the original BFM bases and camera conventions."""

    def __init__(self, data, device='cpu'):
        self.device = str(device)
        for name, key in (('u', 'u'), ('id', 'id'), ('exp', 'exp'),
                          ('u_alb', 'u_alb'), ('alb', 'alb')):
            setattr(self, name, torch.as_tensor(data[key], dtype=torch.float32, device=device))
        self.tri = torch.as_tensor(data['tri'], dtype=torch.int64, device=device)
        self.point_buf = torch.as_tensor(data['point_buf'], dtype=torch.int64, device=device)
        self.persc_proj = torch.tensor([[1015., 0, 0], [0, 1015., 0], [112., 112., 1.]], device=device)
        self.camera_distance = 10.0
        self.init_lit = torch.tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device=device).reshape(1, 1, -1)
        self.SH_a = torch.tensor([np.pi, 2*np.pi/np.sqrt(3.), 2*np.pi/np.sqrt(8.)], dtype=torch.float32, device=device)
        self.SH_c = torch.tensor([1/np.sqrt(4*np.pi), np.sqrt(3.)/np.sqrt(4*np.pi), 3*np.sqrt(5.)/np.sqrt(12*np.pi)], dtype=torch.float32, device=device)

    def compute_shape(self, alpha_id, alpha_exp):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3), face vertice without rotation or translation

        Parameters:
            alpha_id         -- torch.tensor, size (B, 80), identity parameter
            alpha_exp        -- torch.tensor, size (B, 64), expression parameter
        """
        batch_size = alpha_id.shape[0]
        face_shape = torch.einsum('ij,aj->ai', self.id, alpha_id) + torch.einsum('ij,aj->ai', self.exp, alpha_exp) + self.u.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])

    def compute_albedo(self, alpha_alb, normalize=True):
        """
        Return:
            face_albedo     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.), without lighting

        Parameters:
            alpha_alb        -- torch.tensor, size (B, 80), albedo parameter
        """
        batch_size = alpha_alb.shape[0]
        face_albedo = torch.einsum('ij,aj->ai', self.alb, alpha_alb) + self.u_alb.reshape([1, -1])
        if normalize:
            face_albedo = face_albedo / 255.
        return face_albedo.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        v1 = face_shape[:, self.tri[:, 0]]
        v2 = face_shape[:, self.tri[:, 1]]
        v3 = face_shape[:, self.tri[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)

        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_texture(self, face_albedo, face_norm, alpha_sh):
        """
        Return:
            face_texture        -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_albedo         -- torch.tensor, size (B, N, 3), from albedo model, range (0, 1.)
            face_norm           -- torch.tensor, size (B, N, 3), rotated face normal
            alpha_sh            -- torch.tensor, size (B, 27), SH parameter
        """
        batch_size = alpha_sh.shape[0]
        v_num = face_albedo.shape[1]
        a = self.SH_a
        c = self.SH_c
        alpha_sh = alpha_sh.reshape([batch_size, 3, 9])
        alpha_sh = alpha_sh + self.init_lit
        alpha_sh = alpha_sh.permute(0, 2, 1)
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ alpha_sh[..., :1]
        g = Y @ alpha_sh[..., 1:2]
        b = Y @ alpha_sh[..., 2:]
        face_texture = torch.cat([r, g, b], dim=-1) * face_albedo
        return face_texture

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3), pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), use radian
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def split_alpha(self, alpha):
        """
        Return:
            alpha_dict     -- a dict of torch.tensors

        Parameters:
            alpha          -- torch.tensor, size (B, 257)
        """
        alpha_id = alpha[:, :80]
        alpha_exp = alpha[:, 80: 144]
        alpha_alb = alpha[:, 144: 224]
        alpha_a = alpha[:, 224: 227]
        alpha_sh = alpha[:, 227: 254]
        alpha_t = alpha[:, 254:]
        return {
            'id': alpha_id,
            'exp': alpha_exp,
            'alb': alpha_alb,
            'angle': alpha_a,
            'sh': alpha_sh,
            'trans': alpha_t
        }

    def add_directionlight(self, normals, lights):
        '''
        see https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def compute_gray_shading_with_directionlight(self, face_texture, normals):
        '''
        see https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
        '''
        batch_size = normals.shape[0]
        light_positions = torch.tensor(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
        )[None,:,:].expand(batch_size, -1, -1).float()
        light_intensities = torch.ones_like(light_positions).float()*1.7
        lights = torch.cat((light_positions, light_intensities), 2).to(face_texture.device)

        shading = self.add_directionlight(normals, lights)
        texture =  face_texture*shading
        return texture
