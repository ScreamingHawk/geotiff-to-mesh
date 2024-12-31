import argparse
import numpy as np
from osgeo import gdal
from stl import mesh


def process_geotiff_to_mesh(input_file, output_file):
    # Open the GeoTIFF file
    dataset = gdal.Open(input_file)
    if not dataset:
        raise FileNotFoundError(f"Unable to open file: {input_file}")

    # Read raster data as a NumPy array
    band = dataset.GetRasterBand(1)
    elevation_data = band.ReadAsArray()

    # Get the dimensions of the raster
    rows, cols = elevation_data.shape

    # Generate grid points based on the dimensions
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)

    # Flatten arrays for mesh generation
    vertices = np.array([x.flatten(), y.flatten(), elevation_data.flatten()]).T
    vertices[:, 2] = np.nan_to_num(
        vertices[:, 2], nan=0
    )  # Handle missing elevation values

    # Create triangular faces
    faces = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            # Get indices of the quad's corners
            tl = row * cols + col
            tr = tl + 1
            bl = tl + cols
            br = bl + 1
            # Split quad into two triangles
            faces.append([tl, bl, br])
            faces.append([tl, br, tr])

    faces = np.array(faces)

    # Create STL mesh
    mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[f[j], :]

    # Save the mesh as an STL file
    mesh_data.save(output_file)
    print(f"Mesh saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a GeoTIFF to a 3D mesh (STL format)."
    )
    parser.add_argument("input", help="Path to the input GeoTIFF file.")
    parser.add_argument("output", help="Path to the output STL file.")
    args = parser.parse_args()

    try:
        process_geotiff_to_mesh(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
