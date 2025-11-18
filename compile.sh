#!/bin/bash

cd submodules/diff-convex-rasterization/

# # Delete the build, diff_convex_rasterization.egg-info, and dist folders if they exist
rm -rf build
rm -rf diff_convex_rasterization.egg-info

pip install .

cd ..
cd ..