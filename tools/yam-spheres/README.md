# yam-spheres

Collision spheres for the i2rt YAM (6-DoF + linear_4310 gripper) pair on the yam-station, for multi-arm planners
(tools/vamp, tools/curobo). `yam_spheres.json` (`.yml` is the same content) holds:

- `arm_bases`: both bases in the station world frame (bench top z=0): left (-0.3475, +0.305, 0.023), right (-0.3475, -0.305, 0.023), no rotation.
- `chain`: body pos/quat/joint per MJCF body (== URDF link frames; `link_names_urdf` maps names, `gripper` == `link_6`).
- `spheres`: per body, `{center (body frame), radius}`; `spheres_link6_fingers_open/closed` merge the finger spheres into link_6 for planners without finger joints.
- `fit` and `validation_arm_arm`: the numbers below.

Regenerate: `python fit_spheres.py --mjcf <robosimstudio>/assets/embodiments/yam_i2rt/yam_i2rt_bimanual.xml`, then
`python build_and_validate.py --station-mjcf <twin>/assets/yam_station_arms.xml` (needs mujoco, trimesh, scipy).
