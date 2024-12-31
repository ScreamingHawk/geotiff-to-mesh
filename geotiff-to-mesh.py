import argparse
import numpy as np
from osgeo import gdal
from stl import mesh
from tqdm import tqdm
import logging
import os
import glob
from scipy.interpolate import griddata

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enable GDAL exceptions
gdal.UseExceptions()


def try_open_file(file_path):
    """Try to open a file with multiple possible extensions"""
    logger.info(f"Attempting to open: {file_path}")

    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is not None:
        return dataset
    logger.error(f"Could not open {file_path} with GDAL: {gdal.GetLastErrorMsg()}")
    return None


def read_tfw(tiff_path):
    """Read and parse TFW file associated with a TIFF"""
    # Try both .tfw and .tifw extensions
    tfw_path = tiff_path[:-4] + ".tfw"
    if not os.path.exists(tfw_path):
        tfw_path = tiff_path[:-4] + ".tifw"
        if not os.path.exists(tfw_path):
            logger.warning("No world file found")
            return None

    try:
        with open(tfw_path, "r") as f:
            lines = [float(line.strip()) for line in f.readlines()]

        if len(lines) != 6:
            logger.error("Invalid world file format")
            return None

        world_file = {
            "x_pixel_size": lines[0],
            "y_rotation": lines[1],
            "x_rotation": lines[2],
            "y_pixel_size": lines[3],
            "x_top_left": lines[4],
            "y_top_left": lines[5],
        }

        logger.info("World File Contents:")
        logger.info(f"Pixel Size (X): {world_file['x_pixel_size']:.6f}")
        logger.info(f"Rotation Y: {world_file['y_rotation']:.6f}")
        logger.info(f"Rotation X: {world_file['x_rotation']:.6f}")
        logger.info(f"Pixel Size (Y): {world_file['y_pixel_size']:.6f}")
        logger.info(f"Top Left X: {world_file['x_top_left']:.6f}")
        logger.info(f"Top Left Y: {world_file['y_top_left']:.6f}")

        # Calculate approximate coverage area
        dataset = gdal.Open(tiff_path, gdal.GA_ReadOnly)
        if dataset:
            width = dataset.RasterXSize
            height = dataset.RasterYSize

            # Calculate bottom right coordinates
            x_bottom_right = world_file["x_top_left"] + (
                width * world_file["x_pixel_size"]
            )
            y_bottom_right = world_file["y_top_left"] + (
                height * world_file["y_pixel_size"]
            )

            logger.info("\nCoverage Area:")
            logger.info(
                f"Top Left: ({world_file['x_top_left']:.6f}, {world_file['y_top_left']:.6f})"
            )
            logger.info(f"Bottom Right: ({x_bottom_right:.6f}, {y_bottom_right:.6f})")

            # If these appear to be lat/lon coordinates
            if (
                abs(world_file["x_top_left"]) <= 180
                and abs(world_file["y_top_left"]) <= 90
            ):
                logger.info("\nCoordinates appear to be in lat/lon format")
                logger.info(
                    f"Coverage area: approximately {abs(x_bottom_right - world_file['x_top_left']):.2f}° x {abs(y_bottom_right - world_file['y_top_left']):.2f}°"
                )
            else:
                logger.info(
                    "\nCoordinates appear to be in a projected coordinate system (e.g., meters)"
                )
                logger.info(
                    f"Coverage area: approximately {abs(x_bottom_right - world_file['x_top_left']):.2f} x {abs(y_bottom_right - world_file['y_top_left']):.2f} units"
                )

        return world_file

    except Exception as e:
        logger.error(f"Error reading world file: {str(e)}")
        return None


def inspect_geotiff(file_path):
    """Analyze GeoTIFF and its world file if present"""
    # First read the world file if it exists
    world_file = read_tfw(file_path)

    logger.info("\n--- TIFF Analysis ---")
    dataset = try_open_file(file_path)
    if dataset is None:
        logger.error("Could not open file")
        return False

    # Get basic metadata
    logger.info(f"Driver: {dataset.GetDriver().ShortName}")
    logger.info(
        f"Size: {dataset.RasterXSize}x{dataset.RasterYSize}x{dataset.RasterCount}"
    )

    if dataset.RasterCount == 0:
        logger.error("Dataset contains no raster bands")
        return False

    band = dataset.GetRasterBand(1)
    if band is None:
        logger.error("Could not get raster band")
        return False

    # Check metadata
    metadata = band.GetMetadata()
    logger.info("Metadata:")
    for key, value in metadata.items():
        logger.info(f"  {key}: {value}")

    # Check data type
    dtype = gdal.GetDataTypeName(band.DataType)
    logger.info(f"Data type: {dtype}")

    # Get statistics
    stats = band.GetStatistics(True, True)
    logger.info(f"Min: {stats[0]:.2f}")
    logger.info(f"Max: {stats[1]:.2f}")
    logger.info(f"Mean: {stats[2]:.2f}")
    logger.info(f"StdDev: {stats[3]:.2f}")

    # Check color interpretation
    color_interp = band.GetColorInterpretation()
    logger.info(
        f"Color interpretation: {gdal.GetColorInterpretationName(color_interp)}"
    )

    # Check if it has a color table (typical for visual data, not elevation)
    color_table = band.GetColorTable()
    if color_table:
        logger.info("Has color table (likely visual data, not elevation)")

    # Likely elevation data if:
    # 1. Single band
    # 2. Float or Int data type
    # 3. No color table
    # 4. Reasonable elevation range for your area
    is_likely_elevation = (
        dataset.RasterCount == 1
        and dtype in ["Float32", "Float64", "Int16", "Int32"]
        and color_table is None
        and stats[1] - stats[0] < 10000  # Adjust based on expected elevation range
    )

    logger.info(f"Likely contains elevation data: {is_likely_elevation}")
    return is_likely_elevation


def visualize_file_grid(spatial_info):
    """Create an ASCII visualization of how the files are arranged"""
    # Get grid dimensions
    x_coords = [info["x_origin"] for info in spatial_info]
    y_coords = [info["y_origin"] for info in spatial_info]

    # Create grid coordinates - note: y_coords are sorted in reverse
    unique_x = sorted(set(x_coords))
    unique_y = sorted(
        set(y_coords), reverse=True
    )  # Reverse Y-axis to match geographic orientation

    # Create empty grid
    grid = [[" " for _ in range(len(unique_x))] for _ in range(len(unique_y))]

    # Fill grid with file identifiers
    for info in spatial_info:
        x_idx = unique_x.index(info["x_origin"])
        y_idx = unique_y.index(
            info["y_origin"]
        )  # Will now match geographic orientation
        # Extract the numeric suffix from filename
        identifier = os.path.basename(info["file"]).split("_")[-1].split(".")[0]
        grid[y_idx][x_idx] = identifier

    # Debug information
    logger.info("\nCoordinate Information:")
    logger.info(f"X coordinates: {unique_x}")
    logger.info(f"Y coordinates: {unique_y}")

    # Find maximum identifier length for padding
    max_length = max(len(cell) for row in grid for cell in row if cell != " ")

    # Print grid
    logger.info("\nFile Grid Layout:")
    logger.info("+" + "-" * (len(unique_x) * (max_length + 3) - 1) + "+")
    for row in grid:
        logger.info("| " + " | ".join(cell.center(max_length) for cell in row) + " |")
    logger.info("+" + "-" * (len(unique_x) * (max_length + 3) - 1) + "+")


def process_directory_to_mesh(input_dir, output_file, granularity=0.1, scale=0.05):
    """Process all GeoTIFF files in a directory and stitch them into a single mesh

    Args:
        input_dir: Directory containing GeoTIFF files
        output_file: Path to save the output STL file
        granularity: Float between 0 and 1, percentage of points to keep (default: 0.1 = 10%)
        scale: Float, scale factor for the output mesh (default: 0.05 = 5%)
    """
    logger.info(f"Processing directory: {input_dir}")
    logger.info(f"Using granularity: {granularity*100}%")
    logger.info(f"Using scale: {scale*100}%")

    # Find all relevant files
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    logger.info(f"Found {len(tif_files)} TIFF files")

    if not tif_files:
        raise ValueError(f"No TIFF files found in {input_dir}")

    # First pass: collect spatial information
    spatial_info = []
    logger.info("Reading spatial information from files...")
    for tif_file in tqdm(tif_files, desc="[1/5] Reading metadata"):
        dataset = try_open_file(tif_file)
        if dataset is None:
            continue

        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        # Read associated world file if it exists
        world_file = read_tfw(tif_file)

        # If world file exists, use its coordinates instead
        if world_file:
            x_origin = world_file["x_top_left"]
            y_origin = world_file["y_top_left"]
            pixel_width = world_file["x_pixel_size"]
            pixel_height = world_file["y_pixel_size"]
        else:
            x_origin = geotransform[0]
            y_origin = geotransform[3]
            pixel_width = geotransform[1]
            pixel_height = geotransform[5]

        spatial_info.append(
            {
                "file": tif_file,
                "x_origin": x_origin,
                "y_origin": y_origin,
                "pixel_width": pixel_width,
                "pixel_height": pixel_height,
                "projection": projection,
                "geotransform": geotransform,
            }
        )

        dataset = None  # Close dataset

    # Verify all files use the same projection
    projections = {info["projection"] for info in spatial_info}
    if len(projections) > 1:
        logger.warning("Warning: Multiple projections found in dataset!")
        for proj in projections:
            logger.warning(f"Projection found: {proj}")

    # Visualize file grid
    visualize_file_grid(spatial_info)

    # Store all elevation data and their coordinates
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    # Process files
    for info in tqdm(spatial_info, desc="[2/3] Processing TIFF files"):
        dataset = try_open_file(info["file"])
        if dataset is None:
            continue

        band = dataset.GetRasterBand(1)
        elevation_data = band.ReadAsArray()

        # Reduce granularity by taking every Nth point
        step = int(1 / granularity)
        elevation_data = elevation_data[::step, ::step]

        # Generate grid points with reduced granularity
        rows, cols = elevation_data.shape
        x = np.arange(0, dataset.RasterXSize, step)[:cols]
        y = np.arange(0, dataset.RasterYSize, step)[:rows]
        x, y = np.meshgrid(x, y)

        # Transform to world coordinates
        world_x = info["x_origin"] + x * info["pixel_width"]
        world_y = info["y_origin"] + y * info["pixel_height"]

        # Create vertices
        vertices = np.array(
            [world_x.flatten(), world_y.flatten(), elevation_data.flatten()]
        ).T
        vertices = vertices[~np.isnan(elevation_data).flatten()]  # Remove NaN points

        # Generate faces for this tile
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                idx = i * cols + j + vertex_offset
                if not (
                    np.isnan(elevation_data[i, j])
                    or np.isnan(elevation_data[i + 1, j])
                    or np.isnan(elevation_data[i, j + 1])
                    or np.isnan(elevation_data[i + 1, j + 1])
                ):
                    faces.extend(
                        [
                            [idx, idx + cols, idx + cols + 1],
                            [idx, idx + cols + 1, idx + 1],
                        ]
                    )

        all_vertices.extend(vertices)
        all_faces.extend(faces)
        vertex_offset += len(vertices)

        dataset = None

    # Convert to numpy arrays
    all_vertices = np.array(all_vertices)
    all_faces = np.array(all_faces)

    logger.info(
        f"Created mesh with {len(all_vertices):,} vertices and {len(all_faces):,} faces"
    )

    # Scale the vertices
    logger.info(f"Scaling mesh to {scale*100}%")
    all_vertices *= scale

    # Create and save the mesh
    logger.info("Creating final mesh...")
    mesh_data = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(tqdm(all_faces, desc="[3/3] Building mesh")):
        for j in range(3):
            mesh_data.vectors[i][j] = all_vertices[f[j]]

    # Calculate and log mesh statistics
    logger.info("\nMesh Statistics:")
    logger.info(f"Total vertices: {len(all_vertices):,}")
    logger.info(f"Total faces: {len(all_faces):,}")

    # Calculate bounding box
    min_coords = np.min(all_vertices, axis=0)
    max_coords = np.max(all_vertices, axis=0)
    dimensions = max_coords - min_coords

    logger.info("\nBounding Box:")
    logger.info(
        f"X range: {min_coords[0]:.2f} to {max_coords[0]:.2f} ({dimensions[0]:.2f} units)"
    )
    logger.info(
        f"Y range: {min_coords[1]:.2f} to {max_coords[1]:.2f} ({dimensions[1]:.2f} units)"
    )
    logger.info(
        f"Z range: {min_coords[2]:.2f} to {max_coords[2]:.2f} ({dimensions[2]:.2f} units)"
    )

    # Calculate file size estimate
    estimated_size_bytes = len(all_faces) * 50  # Approximate STL binary format size
    estimated_size_mb = estimated_size_bytes / (1024 * 1024)
    logger.info(f"\nEstimated file size: {estimated_size_mb:.1f} MB")

    logger.info(f"\nSaving combined mesh to {output_file}")
    mesh_data.save(output_file)

    # Get actual file size
    actual_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"Actual file size: {actual_size_mb:.1f} MB")
    logger.info("Processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoTIFF files to a 3D mesh (STL format)."
    )
    parser.add_argument(
        "input", help="Path to the input directory containing GeoTIFF files."
    )
    parser.add_argument("output", help="Path to the output STL file.")
    parser.add_argument(
        "--granularity",
        type=float,
        default=0.1,
        help="Percentage of points to keep (0.0-1.0, default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.05,
        help="Scale factor for output mesh (0.0-1.0, default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    if not 0 < args.granularity <= 1:
        parser.error("Granularity must be between 0 and 1")
    if not 0 < args.scale <= 1:
        parser.error("Scale must be between 0 and 1")

    try:
        process_directory_to_mesh(args.input, args.output, args.granularity, args.scale)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
