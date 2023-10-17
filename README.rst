####################################
The optics of the thin-lens-equation
####################################
|TestStatus| |PyPiStatus| |BlackStyle| |PackStyleBlack| |LicenseBadge|

This python-packag is about the thin-lens-equation:

|ImgThinLens|

.. |ImgThinLens| image:: https://github.com/cherenkov-plenoscope/thin_lens/blob/main/readme/thin_lens_bokeh_overview_scale.svg?raw=True

::
     1/f = 1/g + 1/b

The focal-length of the imaging-optics: ``f``.
The image-distance an image has from the aperture's principal plane: ``b``.
The object-distance an object has from the aperture's principal plane: ``g``.
The screen-distance a screen has from the aperture's principal plane: ``s``.

Note that the screen-distance ``s`` is not part of the thin-lens-equation.
In case ``s != b`` the image on the screen is `out of focus', thus blurred.
Only when ``s == b`` the image on the screen is `in focus', thus sharp.

*******
Install
*******

.. code-block:: bash

    pip install thin_lens


*****
Usage
*****

.. code-block:: python

    import thin_lens

    b = thin_lens.image_distance_for_object_distance(
        object_distance=100.0,
        focal_length=1.0,
    )


.. |TestStatus| image:: https://github.com/cherenkov-plenoscope/thin_lens/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/cherenkov-plenoscope/thin_lens/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/thin_lens
    :target: https://pypi.org/project/thin_lens

.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |PackStyleBlack| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack

.. |LicenseBadge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT