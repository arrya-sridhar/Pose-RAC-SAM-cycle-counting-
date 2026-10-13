"""Synthetic robot cell renderer.

A 6-axis arm (xArm6 kinematics, recoloured to the Nachi CMZ05 look: gray base/shoulder, white arm, dark gripper)
rendered with PyBullet's CPU renderer and composited onto the REAL empty Nachi cell photographed in Abhinav's report
(first frame of nachi_9_seq_006).  Returns an RGB frame plus the exact ground-truth robot mask.
"""
import pybullet as p, pybullet_data, numpy as np, cv2, os

HERE = os.path.dirname(os.path.abspath(__file__))
PLATE = os.path.join(HERE, '..', 'assets', 'first_nachi_9_seq_006.png')   # real empty Nachi cell, from Abhinav's report

# appearance presets (albedos chosen so composited intensities match the real frames:
# real wall ~137-145, real white links ~144-174, real gray column ~60-100)
APPEARANCE = {
    'real':    dict(white=0.66, gray=0.36, dark=0.08, plate_gain=1.15),
    'lowcon':  dict(white=0.57, gray=0.40, dark=0.10, plate_gain=1.15),   # white links ~= wall brightness
    'highcon': dict(white=0.92, gray=0.36, dark=0.08, plate_gain=1.00),   # bright white robot, darker wall
    'darkbg':  dict(white=0.80, gray=0.45, dark=0.10, plate_gain=0.35),   # easy: dark background
}


class Scene:
    def __init__(self, W=1280, H=720, cam_eye=(-0.31, -1.35, 0.46), cam_target=(-0.31, 0.0, 0.38), fov=42,
                 appearance='real', light_dir=(0.6, -1.0, 1.6), pedestal=True):
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.rob = p.loadURDF('xarm/xarm6_with_gripper.urdf', [0, 0, 0], useFixedBase=True, physicsClientId=self.cid)
        self.ped = None
        if pedestal:   # static dark pedestal under the robot (NOT part of the GT robot mask)
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.11, 0.11, 0.20], rgbaColor=[0.07, 0.07, 0.075, 1],
                                     physicsClientId=self.cid)
            self.ped = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=[0, 0, -0.201],
                                         physicsClientId=self.cid)
        self.W, self.H = W, H
        self.arm_joints = [1, 2, 3, 4, 5, 6]
        self.grip_joints = [8, 9, 10, 11, 12, 13]
        self.set_camera(cam_eye, cam_target, fov)
        self.light_dir = list(light_dir)
        self.app = APPEARANCE[appearance] if isinstance(appearance, str) else appearance
        self.recolor()

    def set_camera(self, eye, target, fov):
        self.eye, self.target, self.fov = eye, target, fov
        self.view = p.computeViewMatrix(cameraEyePosition=list(eye), cameraTargetPosition=list(target),
                                        cameraUpVector=[0, 0, 1])
        self.proj = p.computeProjectionMatrixFOV(fov=fov, aspect=self.W / self.H, nearVal=0.05, farVal=10)

    def recolor(self):
        a = self.app
        W = [a['white']] * 3 + [1]; G = [a['gray']] * 3 + [1]; D = [a['dark']] * 3 + [1]
        n = p.getNumJoints(self.rob, physicsClientId=self.cid)
        for li in range(-1, n):
            c = G if li in (-1, 0, 1) else (D if li >= 7 else W)
            p.changeVisualShape(self.rob, li, rgbaColor=c, physicsClientId=self.cid)

    def set_q(self, q, grip=0.0):
        for j, v in zip(self.arm_joints, q):
            p.resetJointState(self.rob, j, float(v), physicsClientId=self.cid)
        for j in self.grip_joints:
            p.resetJointState(self.rob, j, float(grip), physicsClientId=self.cid)

    def render(self, q, grip=0.0):
        """returns rgb (H,W,3 uint8, black where nothing rendered), robot mask, pedestal mask"""
        self.set_q(q, grip)
        img = p.getCameraImage(self.W, self.H, self.view, self.proj, renderer=p.ER_TINY_RENDERER, shadow=0,
                               lightDirection=self.light_dir, lightAmbientCoeff=0.62, lightDiffuseCoeff=0.42,
                               lightSpecularCoeff=0.06, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                               physicsClientId=self.cid)
        rgb = np.reshape(np.array(img[2], dtype=np.uint8), (self.H, self.W, 4))[:, :, :3]
        seg = np.reshape(np.array(img[4]), (self.H, self.W))
        obj = np.where(seg >= 0, seg & ((1 << 24) - 1), -1)
        return rgb, obj == self.rob, (obj >= 0) & (obj != self.rob)   # robot mask, all other rendered objects

    def project(self, xyz):
        V = np.array(self.view).reshape(4, 4).T; Pm = np.array(self.proj).reshape(4, 4).T
        v = Pm @ V @ np.array([*xyz, 1.0]); v = v[:3] / v[3]
        return np.array([(v[0] + 1) / 2 * self.W, (1 - v[1]) / 2 * self.H])


def load_plate(W=1280, H=720, gain=1.0):
    im = cv2.imread(PLATE)
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) * gain
    return np.clip(im, 0, 255)


def composite(bg, rgb, fg_mask, blur=1.0, feather=1.0):
    """alpha-composite rendered foreground over background; slight blur to match the soft upscaled plate"""
    fg = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), blur) if blur > 0 else rgb.astype(np.float32)
    a = cv2.GaussianBlur(fg_mask.astype(np.float32), (0, 0), feather)[..., None]
    # un-premultiply the blurred colour at the edges so black background doesn't bleed in
    m = cv2.GaussianBlur(fg_mask.astype(np.float32), (0, 0), blur)[..., None] if blur > 0 else fg_mask[..., None].astype(np.float32)
    fg = fg / np.maximum(m, 1e-3)
    return bg * (1 - a) + np.clip(fg, 0, 255) * a
