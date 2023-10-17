import thin_lens
import numpy as np


def test_focus():
    prng = np.random.Generator(np.random.PCG64(seed=45))
    num_tests = 1000

    for i in range(num_tests):
        f = prng.uniform(low=0.1, high=2)
        o = prng.uniform(low=0.1, high=100)

        b = thin_lens.compute_image_distance_for_object_distance(
            object_distance=o,
            focal_length=f,
        )

        o_back = thin_lens.compute_object_distance_for_image_distance(
            image_distance=b,
            focal_length=f,
        )

        np.testing.assert_almost_equal(o, o_back, decimal=7)


def test_point_transform():
    prng = np.random.Generator(np.random.PCG64(seed=45))
    num_tests = 1000

    for i in range(num_tests):
        f = prng.uniform(low=0.1, high=1.0)

        x = prng.uniform(low=-10, high=+10)
        y = prng.uniform(low=-10, high=+10)
        z = prng.uniform(low=0.1, high=+100)

        point_xyz = [x, y, z]

        point_img = thin_lens.xyz2cxcyb(
            x=point_xyz[0],
            y=point_xyz[1],
            z=point_xyz[2],
            focal_length=f,
        )

        point_xyz_back = thin_lens.cxcyb2xyz(
            cx=point_img[0],
            cy=point_img[1],
            image_distance=point_img[2],
            focal_length=f,
        )

        np.testing.assert_almost_equal(point_xyz, point_xyz_back, decimal=7)
