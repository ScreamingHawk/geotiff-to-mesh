# geotiff-to-mesh

This script converts multiple GeoTIFF files to a single 3D mesh (STL format).

The script was designed to work with files downloaded from Toitu Te Whenua (aka Land Information New Zealand). You can [download maps here](https://data.linz.govt.nz/layer/50767-nz-topo50-maps/).

## Setup

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the conda environment
conda activate geotiff-to-mesh

# Install pre-commit hooks
pre-commit install
```

## Obtaining elevation data from LINZ

You can download elevation data from [Toitu Te Whenua (aka Land Information New Zealand aka LINZ)](https://data.linz.govt.nz/group/national-elevation/data/).

1. Ensure that "National Elevation" is selected as the "Group" on the right hand side
2. Find your desired location and select the map you want
3. From the elevation data panel, select "Export" and then "Zoom to Crop"
4. Move the map to the desired area for exporting
5. Select "Download" to download the GeoTIFF files and metadata

> [!TIP]
> At the time of writing, the elevation data does not display on the map overview.

### Required Files

The downloaded files should include:

- `.tif` files (elevation data)
- `.tfw` files (world files with coordinate information)
- `.xml` files (metadata)

Keep all files in the same directory for processing.

For an example, extract a [7zip](https://www.7-zip.org/) file in the `samples` directory.

## Usage

```bash
# Basic usage (processes directory of TIFFs at 10% granularity, 5% scale)
python geotiff-to-mesh.py input_directory output.stl

# Specify granularity (e.g., 5% of points)
python geotiff-to-mesh.py input_directory output.stl --granularity 0.05

# Specify scale (e.g., 50% of original size)
python geotiff-to-mesh.py input_directory output.stl --scale 0.5

# Enable detailed debug output
python geotiff-to-mesh.py input_directory output.stl --log-level DEBUG
```

### Parameters

- `input_directory`: Directory containing the GeoTIFF files
- `output.stl`: Path for the output STL file
- `--granularity`: Optional. Percentage of points to keep (0.0-1.0, default: 0.1 = 10%)
- `--scale`: Optional. Scale factor for the output mesh (0.0-1.0, default: 0.05 = 5%)
- `--log-level`: Optional. Set logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL, default: INFO)

### Processing Steps

The script will:

1. Process all TIFF files found in the directory
2. Use world file (.tfw) coordinates when available
3. Stitch all elevation data into a single mesh
4. Reduce mesh complexity based on the granularity setting

## Development

This project uses [pre-commit](https://pre-commit.com) hooks to maintain code quality. The Black formatter will automatically run on your Python files when you commit changes.

To manually run formatting on all files:

```bash
pre-commit run --all-files
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
