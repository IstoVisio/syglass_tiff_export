import numpy as np
import sys
import syglass
import tifffile
from tqdm import tqdm
from pathlib import Path
from syglass import pyglass


# Modify these export directories to the desired locations, ending with a slash
IMAGE_EXPORT_DIR = "C:/Users/natha/Downloads/ExportDir/"
MASK_EXPORT_DIR  = "C:/Users/natha/Downloads/ExportDir/"

# Modify these to choose whether to export images, masks, or both
EXPORT_IMAGES = True
EXPORT_MASKS  = True


def export_tiffs(project_path : str):
	if not syglass.is_project(project_path):
		print("\nThe file at the path provided is not a valid syGlass project.")
		return
	
	project = syglass.get_project(project_path)
	timepoint_count = project.get_timepoint_count()
	if timepoint_count > 1:
		print("\nWarning: timeseries projects are not well-supported by this tool.")

	project_name = project.get_name()
	resolution_map = project.get_resolution_map()
	resolution_count = len(resolution_map)

	block_size = project.get_block_size()
	block_count = resolution_map[-1]
	blocks_per_dimension = block_count ** (1.0 / 3.0)
	resolution = (block_size * blocks_per_dimension).astype(np.uint64)

	if EXPORT_IMAGES:
		print("Exporting image data...")
		for z in tqdm(range(resolution[0])):
			slice_prefix = str(z).zfill(7)
			image_slice = project.get_custom_block(0, resolution_count - 1, np.asarray([z, 0, 0]), [1, resolution[1], resolution[2]])
			tifffile.imwrite(IMAGE_EXPORT_DIR + project_name + "_Image_" + slice_prefix + ".tiff", image_slice.data)

	if EXPORT_MASKS:
		print("Exporting mask data...")
		mask_extractor = pyglass.MaskOctreeRasterExtractor(None)
		for z in tqdm(range(resolution[0])):
			slice_prefix = str(z).zfill(7)
			mask_slice = mask_extractor.GetCustomBlock(project.impl, 0, resolution_count - 1, pyglass.vec3(0, 0, float(z)), pyglass.vec3(float(resolution[2]), float(resolution[1]), 1))
			tifffile.imwrite(MASK_EXPORT_DIR + project_name + "_Mask_" + slice_prefix + ".tiff", pyglass.GetRasterAsNumpyArray(mask_slice))

	print("Finished.")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("\nUsage: python syglass_tiff_export.py [path/to/syGlass/file.syg]")
	else:
		export_tiffs(sys.argv[1])