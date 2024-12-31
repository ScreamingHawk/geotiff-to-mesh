# geotiff-to-mesh

This script converts a GeoTIFF file to a 3D mesh (STL format).

The script was designed to work with files downloaded from Toitu Te Whenua (aka Land Information New Zealand). You can [download maps here](https://data.linz.govt.nz/layer/50767-nz-topo50-maps/).

## Setup

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the conda environment
conda activate geotiff-to-mesh
```

## Usage

```bash
python geotiff-to-mesh.py input.tif output.stl
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
