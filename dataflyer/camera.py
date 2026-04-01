"""6DOF fly-through camera with WASD + mouse controls."""

import numpy as np
import glfw


class Camera:
    def __init__(self, position=None, fov=90.0, aspect=16 / 9, near=1e-6, far=1e6):
        self.position = np.array(position if position is not None else [0, 0, 0], dtype=np.float32)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        # Orientation as forward/up vectors (right derived from cross product)
        self._forward = np.array([0, 0, -1], dtype=np.float32)
        self._up = np.array([0, 1, 0], dtype=np.float32)

        # Movement
        self.speed = 1.0  # units/sec, will be auto-scaled from data
        self.mouse_sensitivity = 0.002
        self.roll_speed = 1.5  # rad/sec

        # Input state
        self._keys = set()
        self._last_cursor = None
        self._mouse_captured = False
        self._moving = False

    @property
    def forward(self):
        return self._forward / np.linalg.norm(self._forward)

    @property
    def right(self):
        r = np.cross(self.forward, self._up)
        return r / np.linalg.norm(r)

    @property
    def up(self):
        u = np.cross(self.right, self.forward)
        return u / np.linalg.norm(u)

    def view_matrix(self):
        """Returns 4x4 view matrix (world -> camera)."""
        f = self.forward
        r = self.right
        u = self.up
        p = self.position

        m = np.eye(4, dtype=np.float32)
        m[0, :3] = r
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(r, p)
        m[1, 3] = -np.dot(u, p)
        m[2, 3] = np.dot(f, p)
        return m

    def projection_matrix(self):
        """Returns 4x4 perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(self.fov) / 2)
        n, fa = self.near, self.far
        a = self.aspect

        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / a
        m[1, 1] = f
        m[2, 2] = (fa + n) / (n - fa)
        m[2, 3] = (2 * fa * n) / (n - fa)
        m[3, 2] = -1
        return m

    def update(self, dt):
        """Process input and update position/orientation. Returns True if camera moved or rotated."""
        # Don't reset _moving here -- mouse rotation sets it during poll_events()
        moved = self._moving
        self._moving = False
        velocity = np.zeros(3, dtype=np.float32)

        if glfw.KEY_W in self._keys:
            velocity += self.forward
        if glfw.KEY_S in self._keys:
            velocity -= self.forward
        if glfw.KEY_A in self._keys:
            velocity -= self.right
        if glfw.KEY_D in self._keys:
            velocity += self.right
        if glfw.KEY_SPACE in self._keys:
            velocity += self.up
        if glfw.KEY_LEFT_SHIFT in self._keys:
            velocity -= self.up

        if np.dot(velocity, velocity) > 0:
            velocity = velocity / np.linalg.norm(velocity) * self.speed * dt
            self.position += velocity
            self._moving = True

        # Roll
        if glfw.KEY_Q in self._keys:
            self._roll(-self.roll_speed * dt)
            self._moving = True
        if glfw.KEY_E in self._keys:
            self._roll(self.roll_speed * dt)
            self._moving = True

        moved = moved or self._moving
        return moved

    @property
    def is_moving(self):
        return self._moving

    def on_key(self, key, action):
        if action == glfw.PRESS:
            self._keys.add(key)
        elif action == glfw.RELEASE:
            self._keys.discard(key)

    def on_mouse_button(self, button, action):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_captured = action == glfw.PRESS

    def on_cursor(self, xpos, ypos):
        if self._last_cursor is None:
            self._last_cursor = (xpos, ypos)
            return

        if not self._mouse_captured:
            self._last_cursor = (xpos, ypos)
            return

        dx = xpos - self._last_cursor[0]
        dy = ypos - self._last_cursor[1]
        self._last_cursor = (xpos, ypos)

        if abs(dx) > 0 or abs(dy) > 0:
            self._yaw(-dx * self.mouse_sensitivity)
            self._pitch(-dy * self.mouse_sensitivity)
            self._moving = True

    def on_scroll(self, offset):
        self.speed *= 1.15 ** offset

    def _yaw(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        u = self.up
        # Rodrigues rotation of forward around up
        self._forward = (
            self._forward * c + np.cross(u, self._forward) * s + u * np.dot(u, self._forward) * (1 - c)
        )

    def _pitch(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        r = self.right
        self._forward = (
            self._forward * c + np.cross(r, self._forward) * s + r * np.dot(r, self._forward) * (1 - c)
        )
        self._up = (self._up * c + np.cross(r, self._up) * s + r * np.dot(r, self._up) * (1 - c))

    def _roll(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        f = self.forward
        self._up = self._up * c + np.cross(f, self._up) * s + f * np.dot(f, self._up) * (1 - c)

    def auto_scale(self, positions, masses=None, boxsize=None):
        """Set speed and clip planes. Starts at the box center looking along -z."""
        pmin = positions.min(axis=0)
        pmax = positions.max(axis=0)
        extent = np.linalg.norm(pmax - pmin)

        if boxsize is not None:
            center = np.ones(3, dtype=np.float32) * boxsize / 2
        else:
            center = ((pmin + pmax) / 2).astype(np.float32)

        self.position = center + np.array([0, 0, extent * 0.3], dtype=np.float32)
        self.speed = extent / 10
        self.near = extent * 1e-6
        self.far = extent * 10
