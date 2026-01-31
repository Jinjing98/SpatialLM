
#/////////Locally on laptop
# (base) jinjingxu@G27LP0076-Linux:~$ rerun /home/jinjingxu/hpc_Projects/VLM_3D/SpatialLM/outputs/scene40753679.rrd
# [2026-01-31T08:40:24Z WARN  re_log_encoding::decoder] Found log stream with Rerun version 0.21.0, which is incompatible with the local Rerun version 0.22.1. Loading will try to continue, but might fail in subtle ways.
# [2026-01-31T08:40:24Z INFO  winit::platform_impl::linux::x11::window] Guessed window scale factor: 1
# [2026-01-31T08:40:25Z WARN  wgpu_hal::gles::adapter] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.
# [2026-01-31T08:40:25Z WARN  wgpu_hal::gles::adapter] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.


# ////////////Visualize the point cloud and the predicted layout
# notice it is on local machine (it need the spatiallm env been activated and rrd is snapped)
# sudo mount --all
# (base) jinjingxu@G27LP0076-Linux:~$ source activate /mnt/cluster/environments/jinjingxu/pkg/envs/spatiallm/
# (spatiallm) jinjingxu@G27LP0076-Linux:~$ rerun /tmp/test.rrd 

# rerun /mnt/cluster/workspaces/jinjingxu/proj/vlm/SpatialLM/outputs/scene40753679.rrd