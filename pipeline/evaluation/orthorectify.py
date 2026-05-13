import math
import logging
import rasterio
from pathlib import Path
from pyproj import Transformer
from osgeo import gdal

logger = logging.getLogger(__name__)

TARGET_GSD = 4.75
MAX_ORTHO_DIM = 15000
MAX_AREA_RATIO = 8.0


def _load_rpc_dict(txt_path: Path) -> dict:
    rpc = {}
    for line in open(txt_path, "r"):
        if ':' in line:
            k, v = map(str.strip, line.split(":", 1))
            rpc[k] = v if 'COEFF' not in k else " ".join(v.split())
    return rpc

def _is_rpc_sane(ds, rpc_dict: dict) -> bool:
    """Sanity check for RPC parameters to avoid huge ortho outputs."""
    w, h = ds.RasterXSize, ds.RasterYSize
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_HEIGHT={rpc_dict.get('HEIGHT_OFF', 0)}"])
    
    # Transformation of the four corners of the image
    pts = [tr.TransformPoint(False, x, y) for x, y in [(0,0), (w,0), (w,h), (0,h)]]
    if not all(success for success, _ in pts):
        return False
        
    lons, lats = zip(*[(p[0], p[1]) for _, p in pts])
    if any(math.isnan(v) or abs(lats[i])>90 or abs(lons[i])>180 for i, v in enumerate(lons)):
        return False

    # Estimate metric dimensions
    meters_x = (max(lons) - min(lons)) * 111320.0 * math.cos(math.radians(sum(lats)/4))
    meters_y = (max(lats) - min(lats)) * 111320.0
    ow, oh = meters_x / TARGET_GSD, meters_y / TARGET_GSD

    if ow <= 0 or oh <= 0 or ow > MAX_ORTHO_DIM or oh > MAX_ORTHO_DIM:
        return False
        
    return (ow * oh) / (w * h) <= MAX_AREA_RATIO


def run_orthorectify(phisat_tiff: Path, dem_path: Path, rpc_path: Path, output_path: Path):
    for p in [phisat_tiff, dem_path, rpc_path]:
        if not p.exists(): raise FileNotFoundError(f"File not found: {p}")

    with rasterio.open(dem_path) as dem:
        cx, cy = dem.bounds.left + dem.bounds.right, dem.bounds.bottom + dem.bounds.top
        cx, cy = cx / 2.0, cy / 2.0
        if dem.crs != "EPSG:4326":
            cx, cy = Transformer.from_crs(dem.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
            
    zone = int((cx + 180) // 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone

    rpc = _load_rpc_dict(rpc_path)
    vrt_ds = gdal.Translate('', str(phisat_tiff), format='VRT')
    vrt_ds.SetMetadata(rpc, 'RPC')

    if not _is_rpc_sane(vrt_ds, rpc):
        logger.warning(f"Preskakujem {phisat_tiff.name}: Wrong RPC parameters.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdal.Warp(
        str(output_path), vrt_ds,
        format="GTiff",
        xRes=TARGET_GSD, yRes=TARGET_GSD,
        dstSRS=f"EPSG:{epsg}",
        resampleAlg=gdal.GRA_Cubic,
        transformerOptions=[f"RPC_DEM={dem_path}"],
        rpc=True,
        dstNodata=0,
        multithread=True
    )
    
    logger.info(f"Successfully created: {output_path.name}")
    return output_path