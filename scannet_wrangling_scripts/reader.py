import argparse
from concurrent.futures import process
import os, sys
from tqdm import tqdm
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--scans_folder',default='../scannet/scans', help='dataset root') #  required=True
parser.add_argument('--scan_list_file', required=False, default=None, help='scan list file')
parser.add_argument('--single_debug_scan_id', required=False, default=None, help='single scan to debug')
parser.add_argument('--output_path', default='../scannet/scans', help='path to output folder') # required=True
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', default=True, dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', default=True, dest='export_intrinsics', action='store_true')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--rgb_resize', nargs='+', type=int, default=[224, 224], help='width height')
parser.add_argument('--depth_resize', nargs='+', type=int, default=[224, 224], help='width height')
parser.set_defaults(export_depth_images=True, export_color_images=True, export_poses=True, export_intrinsics=True)

opt = parser.parse_args()
print(opt)

def process_scan(opt, scan_job, count=None, progress=None):
  filename = scan_job[0]
  output_path = scan_job[1]
  scan_name = scan_job[2]
  # if filename exists
  if not os.path.exists(filename):
    # print(f"File {filename} does not exist, skipping...")
    return

  if not os.path.exists(output_path):
      os.makedirs(output_path)
  # load the data
  sys.stdout.write('loading %s...' % opt.scans_folder)
  sd = SensorData(filename)
  sys.stdout.write('loaded!\n')
  if opt.export_depth_images:
    sd.export_depth_images(os.path.join(output_path, 'sensor_data'), image_size=opt.depth_resize)
  if opt.export_color_images:
    sd.export_color_images(os.path.join(output_path, 'sensor_data'), image_size=opt.rgb_resize)
  if opt.export_poses:
    sd.export_poses(os.path.join(output_path, 'sensor_data'))
  if opt.export_intrinsics:
    sd.export_intrinsics(output_path, scan_name)

  if progress is not None:
    progress.value += 1
    print(f"Completed scan {filename}, {progress.value} of total {count}.")

def main():


  if opt.single_debug_scan_id is not None:
    scans = [opt.single_debug_scan_id]
  elif opt.scan_list_file is not None:
    with open(opt.scan_list_file, "r") as f:
      scans = f.readlines()
    scans = [scan.strip() for scan in scans]
  else:
    scans = sorted([d for d in os.listdir(opt.scans_folder) if os.path.isdir(os.path.join(opt.scans_folder, d))])
  
  
  input_files = [os.path.join(opt.scans_folder, f"{scan}/{scan}.sens") for scan in scans]

  output_dirs = [os.path.join(opt.output_path, scan) for scan in scans]

  scan_jobs = list(zip(input_files, output_dirs, scans))

  if opt.num_workers == 1:
    for scan_job in tqdm(scan_jobs):
      process_scan(opt, scan_job)
  else:

    pool = Pool(opt.num_workers)
    manager = Manager()

    count = len(scan_jobs)
    progress = manager.Value('i', 0)


    pool.map(
          partial(
              process_scan,
              opt,
              count=count,
              progress=progress
          ),
          scan_jobs,
    )

if __name__ == '__main__':
    main()