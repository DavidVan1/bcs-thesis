import math
import logging
import rasterio
from pathlib import Path
from pyproj import Transformer
from osgeo import gdal

logger = logging.getLogger(__name__)

TARGET_GSD = 4.75
MAX_ORTHO_DIM = 25000
MAX_AREA_RATIO = 8.0

def _get_clean_rpc_dict(txt_path: Path) -> dict:
    raw_rpc = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = map(str.strip, line.split(":", 1))
            
            if 'COEFF' in k:
                raw_rpc[k] = [float(x) for x in v.split()]
            else:
                try:
                    raw_rpc[k] = float(v)
                except ValueError:
                    pass

    def _scalar(key):
        return str(float(raw_rpc[key]))

    def _coeff(key):
        return " ".join(str(float(v)) for v in raw_rpc[key])

    return {
        "LINE_OFF": _scalar("LINE_OFF"),
        "SAMP_OFF": _scalar("SAMP_OFF"),
        "LAT_OFF": _scalar("LAT_OFF"),
        "LONG_OFF": _scalar("LONG_OFF"),
        "HEIGHT_OFF": _scalar("HEIGHT_OFF"),
        "LINE_SCALE": _scalar("LINE_SCALE"),
        "SAMP_SCALE": _scalar("SAMP_SCALE"),
        "LAT_SCALE": _scalar("LAT_SCALE"),
        "LONG_SCALE": _scalar("LONG_SCALE"),
        "HEIGHT_SCALE": _scalar("HEIGHT_SCALE"),
        "LINE_NUM_COEFF": _coeff("LINE_NUM_COEFF"),
        "LINE_DEN_COEFF": _coeff("LINE_DEN_COEFF"),
        "SAMP_NUM_COEFF": _coeff("SAMP_NUM_COEFF"),
        "SAMP_DEN_COEFF": _coeff("SAMP_DEN_COEFF"),
    }

def _validate_rpc_extent(ds, rpc_dict: dict) -> bool:
    """Checks if the RPC parameters are reasonable for orthorectification."""
    w, h = ds.RasterXSize, ds.RasterYSize
    tr = gdal.Transformer(ds, None, ["METHOD=RPC", f"RPC_HEIGHT={rpc_dict.get('HEIGHT_OFF', 0)}"])
    
    pts = [tr.TransformPoint(False, x, y) for x, y in [(0,0), (w,0), (w,h), (0,h)]]
    if not all(success for success, _ in pts):
        return False
        
    lons, lats = zip(*[(p[0], p[1]) for _, p in pts])
    if any(math.isnan(v) or abs(lats[i])>90 or abs(lons[i])>180 for i, v in enumerate(lons)):
        return False

    meters_x = (max(lons) - min(lons)) * 111320.0 * math.cos(math.radians(sum(lats)/4))
    meters_y = (max(lats) - min(lats)) * 111320.0
    ow, oh = meters_x / TARGET_GSD, meters_y / TARGET_GSD

    if ow <= 0 or oh <= 0 or ow > MAX_ORTHO_DIM or oh > MAX_ORTHO_DIM:
        return False
        
    return (ow * oh) / (w * h) <= MAX_AREA_RATIO


def run_orthorectify(phisat_tiff: Path, dem_path: Path, rpc_path: Path, output_path: Path):
    for p in [phisat_tiff, dem_path, rpc_path]:
        if not p.exists(): 
            raise FileNotFoundError(f"File not found: {p}")

    with rasterio.open(dem_path) as dem:
        cx, cy = dem.bounds.left + dem.bounds.right, dem.bounds.bottom + dem.bounds.top
        cx, cy = cx / 2.0, cy / 2.0
        if dem.crs.to_string().upper() != "EPSG:4326":
            cx, cy = Transformer.from_crs(dem.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
            
    zone = int((cx + 180) // 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone

    rpc = _get_clean_rpc_dict(rpc_path)
    
    vrt_ds = gdal.Translate('', str(phisat_tiff), format='VRT')
    vrt_ds.SetMetadata(rpc, 'RPC')

    if not _validate_rpc_extent(vrt_ds, rpc):
        logger.warning(f"Skipping {phisat_tiff.name}: Wrong RPC parameters.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdal.Warp(
        str(output_path), 
        vrt_ds,
        format="GTiff",
        xRes=TARGET_GSD, 
        yRes=TARGET_GSD,
        dstSRS=f"EPSG:{epsg}",
        resampleAlg=gdal.GRA_Cubic,
        transformerOptions=[f"RPC_DEM={dem_path}"],
        rpc=True,
        dstNodata=0,
        targetAlignedPixels=True,
        multithread=True,
        warpOptions=["NUM_THREADS=ALL_CPUS"]
    )
    
    vrt_ds = None 
    
    logger.info(f"Successfully created: {output_path.name}")
    return output_path