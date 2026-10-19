import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.transforms import offset_copy
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.ndimage import (gaussian_filter, maximum_filter, minimum_filter,
                           uniform_filter)

TAG = '20160119-20160126'
CENTER_LON = 120
PROJ = ccrs.LambertConformal(central_longitude=CENTER_LON, central_latitude=30,
                             standard_parallels=(15, 45))
# data longitudes are shifted by -CENTER_LON so that the dateline is not crossed
DATA_CRS = ccrs.PlateCarree(central_longitude=CENTER_LON)
# rectangular map extent in PROJ coords (m), like the CWA analysis charts;
# corners ~ (86E, 9S), (63E, 44N), (177E, 44N), (154E, 9S)
EXTENT = [-4.4e6, 4.4e6, -3.9e6, 2.6e6]
FIGSIZE = (16.5, 11.7)               # A3 landscape
DPI = 150

KT = 1.94384                         # m/s -> knot
BARB_STRIDE = 12                     # 12 grid points = 3 deg
BARB_KW = dict(color='navy', length=5.5, linewidth=0.6)

SIGMA = {'msl': 2, 'z': 2, 't': 2, 'theta_e': 1.5, 'd': 4, 'tdd': 1}
LEVEL_IDX = {'sfc': 0, '850': 1, '500': 2, '200': 3}   # index in pressure_level
LEVEL_HPA = {'sfc': 1000, '850': 850, '500': 500, '200': 200}

# Layers drawn on each map, in any order (draw order is fixed by zorder).
# Available: 'msl' (sea-level pressure, H/L), 'height' (geopotential height, H/L),
# 'temp' (isotherms, W/C), 'theta_e' (shaded), 'wet' (T-Td <= 3 hatch),
# 'isotach' (shaded wind speed), 'div' (divergence contours), 'barbs' (wind barbs)
LAYERS = {
    'sfc': ['msl', 'barbs'],
    '850': ['theta_e', 'wet', 'height', 'temp', 'barbs'],
    '500': ['height', 'temp', 'barbs'],
    '200': ['isotach', 'height', 'div', 'barbs'],
}
# Per-level style: height/temp contour step, labelled values (h_lab_every: also label
# multiples of it), H/L threshold (gpm), height label color
LEVEL_STYLE = {
    'sfc': dict(h_step=30, h_lab=[], h_lab_every=120, h_col='red', t_step=3,
                t_lab=[0, 15, 30], h_thresh=20),
    '850': dict(h_step=30, h_lab=[1500], h_lab_every=None, h_col='red', t_step=3,
                t_lab=[-15, 0, 15], h_thresh=20),
    '500': dict(h_step=60, h_lab=[5760, 5880], h_lab_every=None, h_col='red', t_step=5,
                t_lab=[-10, -40], h_thresh=20),
    '200': dict(h_step=120, h_lab=[], h_lab_every=240, h_col='k', t_step=5,
                t_lab=[-50, -60], h_thresh=40),
}

# zorder: shading < wet hatch < isotherms < height < barbs < letters
Z_SHADE, Z_WET, Z_TEMP, Z_HGT, Z_BARB, Z_LETTER = 1, 2, 3, 4, 5, 6


def smooth(field, sigma):
    return gaussian_filter(field, sigma, mode='nearest')


# ---------------- thermodynamics (Bolton 1980) ----------------
def vapour_pressure(q, p):
    return np.maximum(q * p / (0.622 + 0.378 * q), 1e-3)

def dewpoint(q, p):
    x = np.log(vapour_pressure(q, p) / 6.112)
    return 243.5 * x / (17.67 - x)          # deg C

def theta_e(t, q, p):
    e = vapour_pressure(q, p)
    r = 1000 * q / (1 - q)
    tl = 2840 / (3.5 * np.log(t) - np.log(e) - 4.805) + 55
    return (t * (1000 / p) ** (0.2854 * (1 - 0.28e-3 * r))
            * np.exp((3.376 / tl - 0.00254) * r * (1 + 0.81e-3 * r)))


# ---------------- map helpers ----------------
def make_axes():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_extent(EXTENT, crs=PROJ)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m'),
                   facecolor='0.96', edgecolor='none', zorder=0)
    ax.coastlines('50m', color='0.35', linewidth=0.6, zorder=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                      color='0.75', linestyle='--', x_inline=False, y_inline=False)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-10, 71, 10))
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 9}
    return fig, ax


def set_titles(ax, left, valid):
    ax.set_title(left, loc='left', fontsize=12 if len(left) <= 85 else 10 if len(left) <= 90 else 9 if len(left) <= 96 else 8, pad=8)
    ax.set_title(f'ERA5   Valid: {valid:%Y-%m-%d %H}UTC', loc='right', fontsize=13, pad=8)


def draw_barbs(ax, x, y, u, v):
    s = slice(None, None, BARB_STRIDE)
    ax.barbs(x[s], y[s], u[s, s] * KT, v[s, s] * KT, transform=DATA_CRS,
             zorder=Z_BARB, **BARB_KW)


def mark_extrema(ax, x, y, field, size, threshold, kind, letter, color,
                 fmt=None, fontsize=22, min_lat=None):
    """Mark local max/min of a smoothed field with a letter (and value)."""
    filt = maximum_filter if kind == 'max' else minimum_filter
    ext = filt(field, size=size, mode='nearest')
    dev = np.abs(field - uniform_filter(field, size=size, mode='nearest'))
    m = (field == ext) & (dev > threshold)
    h = size // 2
    m[:h], m[-h:], m[:, :h], m[:, -h:] = False, False, False, False
    if min_lat is not None:
        m[np.asarray(y) < min_lat] = False
    tr = DATA_CRS._as_mpl_transform(ax)
    for j, i in zip(*np.nonzero(m)):
        # skip points outside the rectangular map (text clipping is unreliable)
        px, py = PROJ.transform_point(x[i], y[j], DATA_CRS)
        if not (EXTENT[0] < px < EXTENT[1] and EXTENT[2] < py < EXTENT[3]):
            continue
        ax.text(x[i], y[j], letter, transform=DATA_CRS, color=color, fontsize=fontsize,
                fontweight='bold', ha='center', va='center', clip_on=True, zorder=Z_LETTER)
        if fmt is not None:
            ax.text(x[i], y[j], fmt(field[j, i]), color=color, fontsize=11, fontweight='bold',
                    ha='center', va='top', clip_on=True, zorder=Z_LETTER,
                    transform=offset_copy(tr, fig=ax.figure, y=-10, units='points'))


def contour_labeled(ax, x, y, f, levels, lw, color, label_levels, label_color,
                    ls='solid', zorder=3, fontsize=10):
    cs = ax.contour(x, y, f, levels=levels, colors=color, linewidths=lw,
                    linestyles=ls, transform=DATA_CRS, zorder=zorder)
    lab = [l for l in label_levels if l in cs.levels]
    if lab:
        ax.clabel(cs, levels=lab, fmt='%d', colors=label_color, inline=True,
                  fontsize=fontsize)
    return cs


def multiples(lo, hi, step):
    return np.arange(np.ceil(lo / step) * step, hi + 1e-6, step)


# ---------------- individual maps ----------------
class Ctx:
    """Plot context of one map: grid, level, style and lazily loaded fields."""
    def __init__(self, x, y, lev, sfc, up):
        self.x, self.y, self.lev, self.p = x, y, lev, LEVEL_HPA[lev]
        self.style = LEVEL_STYLE[lev]
        self.handles = []                  # legend handles collected by layers
        self._src = {'msl': sfc}           # msl from the sfc file, the rest from upper
        self._up, self._cache = up, {}

    def f(self, name):
        if name not in self._cache:
            src = self._src.get(name, self._up)
            self._cache[name] = src[name].values.astype(float)
        return self._cache[name]


# ---------------- layers: layer_xxx(ax, ctx) -> (title fragment, (mappable, label) | None) ----
def layer_msl(ax, c):
    p = smooth(c.f('msl'), SIGMA['msl']) / 100
    levels = multiples(p.min(), p.max(), 4)
    lw = [1.8 if l % 20 == 0 else 0.9 for l in levels]
    cs = ax.contour(c.x, c.y, p, levels=levels, colors='k', linewidths=lw,
                    transform=DATA_CRS, zorder=Z_HGT)
    lab = [l for l in levels if l % 20 == 0 and l in cs.levels]
    ax.clabel(cs, levels=lab, fmt='%d', colors='red', inline=True, fontsize=10)
    f = lambda a: f'{int(round(a / 2) * 2)}'
    mark_extrema(ax, c.x, c.y, p, 41, 2, 'max', 'H', 'blue', f)
    mark_extrema(ax, c.x, c.y, p, 41, 2, 'min', 'L', 'red', f)
    return 'MSLP (hPa)', None


def layer_height(ax, c):
    st = c.style
    gpm = smooth(c.f('z') / 9.80665, SIGMA['z'])
    lv = multiples(gpm.min(), gpm.max(), st['h_step'])
    lab = list(st['h_lab'])
    if st['h_lab_every']:
        lab += [l for l in lv if l % st['h_lab_every'] == 0]
    contour_labeled(ax, c.x, c.y, gpm, lv, 1.0, 'k', lab, st['h_col'], zorder=Z_HGT)
    mark_extrema(ax, c.x, c.y, gpm, 41, st['h_thresh'], 'max', 'H', 'blue', lambda a: f'{a:.0f}')
    mark_extrema(ax, c.x, c.y, gpm, 41, st['h_thresh'], 'min', 'L', 'red', lambda a: f'{a:.0f}')
    return 'Height (gpm)', None


def layer_temp(ax, c):
    st = c.style
    tc = smooth(c.f('t'), SIGMA['t']) - 273.15
    contour_labeled(ax, c.x, c.y, tc, multiples(tc.min(), tc.max(), st['t_step']), 1.0,
                    'darkred', st['t_lab'], 'red', ls='dashed', zorder=Z_TEMP)
    # W/C only outside the flat tropical temperature field
    mark_extrema(ax, c.x, c.y, tc, 61, 3, 'max', 'W', 'red', min_lat=15)
    mark_extrema(ax, c.x, c.y, tc, 61, 3, 'min', 'C', 'blue', min_lat=15)
    return 'Temp (°C, dashed)', None


def layer_theta_e(ax, c):
    # theta_e shaded only for >= 339 K, every 3 K
    the = smooth(theta_e(c.f('t'), c.f('q'), c.p), SIGMA['theta_e'])
    cf = ax.contourf(c.x, c.y, the, levels=np.arange(339, 364, 3), cmap='YlOrRd', alpha=0.5,
                     extend='max', transform=DATA_CRS, zorder=Z_SHADE)
    return 'θe ≥ 339 K (shaded)', (cf, 'θe (K)')


def layer_wet(ax, c):
    tdd = smooth(c.f('t') - (dewpoint(c.f('q'), c.p) + 273.15), SIGMA['tdd'])
    with matplotlib.rc_context({'hatch.color': 'green', 'hatch.linewidth': 0.8}):
        wet = ax.contourf(c.x, c.y, tdd, levels=[-50, 3], colors='none', hatches=['...'],
                          transform=DATA_CRS, zorder=Z_WET)
    wet.set_edgecolor('green')
    wet.set_linewidth(0)
    ax.contour(c.x, c.y, tdd, levels=[3], colors='green', linewidths=0.6,
               transform=DATA_CRS, zorder=Z_WET)
    c.handles.append(mpatches.Patch(facecolor='none', edgecolor='green', hatch='...',
                                    linewidth=0, label='Wet area (T$-$Td ≤ 3°C)'))
    return None, None                      # shown in the legend instead of the title


def layer_isotach(ax, c):
    # isotachs >= 60 kt shaded green, every 20 kt
    spd = np.hypot(c.f('u'), c.f('v')) * KT
    lv_spd = np.arange(60, 221, 20)
    greens = plt.get_cmap('Greens')(np.linspace(0.2, 0.9, len(lv_spd)))   # skip near-white
    cf = ax.contourf(c.x, c.y, spd, levels=lv_spd, colors=greens[:-1], alpha=0.6, extend='max',
                     transform=DATA_CRS, zorder=Z_SHADE)
    cf.cmap.set_over(greens[-1])
    return 'Isotach ≥ 60 kt (shaded)', (cf, 'Wind speed (kt)')


def layer_div(ax, c):
    div = smooth(c.f('d'), SIGMA['d']) * 1e5
    for lvls, col, ls in (([1, 2, 3, 4, 6, 8], 'red', 'solid'),
                          ([-8, -6, -4, -3, -2, -1], 'blue', 'dashed')):
        cs = ax.contour(c.x, c.y, div, levels=lvls, colors=col, linewidths=0.9,
                        linestyles=ls, transform=DATA_CRS, zorder=Z_TEMP)
        lab = [l for l in lvls if l in cs.levels]
        if lab:
            ax.clabel(cs, levels=lab, fmt='%d', colors=col, inline=True, fontsize=8)
    return 'Div (10⁻⁵ s⁻¹)', None


def layer_barbs(ax, c):
    draw_barbs(ax, c.x, c.y, c.f('u'), c.f('v'))
    return 'Wind (kt)', None


LAYER_FUNCS = {'msl': layer_msl, 'height': layer_height, 'temp': layer_temp,
               'theta_e': layer_theta_e, 'wet': layer_wet, 'isotach': layer_isotach,
               'div': layer_div, 'barbs': layer_barbs}


def plot_level(ax, c):
    """Draw LAYERS[level] on ax; returns the left title."""
    names = LAYERS[c.lev]
    bad = [n for n in names if n not in LAYER_FUNCS]
    if bad:
        raise ValueError(f"Unknown layer(s) {bad} for level '{c.lev}'; "
                         f"available: {list(LAYER_FUNCS)}")
    frags, cbars = [], []
    for n in names:
        title, cb = LAYER_FUNCS[n](ax, c)
        if title:
            frags.append(title)
        if cb:
            cbars.append(cb)
    # one horizontal colorbar per shaded layer, stacked at the bottom
    for i, (mappable, label) in enumerate(cbars):
        if i == 0:
            cb = ax.figure.colorbar(mappable, ax=ax, orientation='horizontal', pad=0.06,
                                    shrink=0.6, aspect=40)
            cb.ax.apply_aspect()
            pos = cb.ax.get_position()
        else:                              # next ones: same size, below the previous one
            pos = pos.translated(0, -(pos.height + 0.06))
            cb = ax.figure.colorbar(mappable, cax=ax.figure.add_axes(pos),
                                    orientation='horizontal')
        cb.set_label(label)
    if c.handles:
        ax.legend(handles=c.handles, loc='lower left', fontsize=11, framealpha=0.9).set_zorder(10)
    # sfc: MSLP is not on 1000 hPa, so the prefix is dropped when 'msl' is drawn
    prefix = '' if c.lev == 'sfc' and 'msl' in names else f'{c.p} hPa '
    return prefix + ', '.join(frags)


# ---------------- driver ----------------
def plot_time(args):
    data_dir, outdir, levels, tstr = args
    t0 = time.time()
    valid = np.datetime64(f'{tstr[:4]}-{tstr[4:6]}-{tstr[6:8]}T{tstr[8:]}:00')
    sfc = xr.open_dataset(Path(data_dir) / f'{TAG}_sfc.nc')
    up = xr.open_dataset(Path(data_dir) / f'{TAG}_upper.nc')
    it = int(np.nonzero(sfc.valid_time.values == valid)[0][0])
    assert up.valid_time.values[it] == valid
    x = up.longitude.values - CENTER_LON
    y = up.latitude.values
    ts = (valid.astype('datetime64[s]').astype(object))
    done = []
    for lev in levels:
        # load only the needed level and variables for this time
        ctx = Ctx(x, y, lev, sfc.isel(valid_time=it),
                  up.isel(valid_time=it, pressure_level=LEVEL_IDX[lev]))
        fig, ax = make_axes()
        title = plot_level(ax, ctx)
        set_titles(ax, title, ts)
        out = Path(outdir) / lev / f'{tstr}_{lev}.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        done.append(str(out))
    sfc.close(); up.close()
    return tstr, done, time.time() - t0


def main(data_dir, outdir, levels, times, workers):
    with xr.open_dataset(Path(data_dir) / f'{TAG}_sfc.nc') as ds:
        all_times = [f'{t:%Y%m%d%H}' for t in ds.valid_time.to_index()]
    times = times or all_times
    jobs = [(data_dir, outdir, levels, t) for t in times]
    if workers <= 1:
        results = map(plot_time, jobs)
        for tstr, done, dt in results:
            print(f'{tstr}: {len(done)} figs, {dt:.1f}s ({dt / max(len(done), 1):.1f}s/fig)', flush=True)
    else:
        with ProcessPoolExecutor(workers) as ex:
            for tstr, done, dt in ex.map(plot_time, jobs):
                print(f'{tstr}: {len(done)} figs, {dt:.1f}s ({dt / max(len(done), 1):.1f}s/fig)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./era5_data')
    parser.add_argument('--outdir', type=str, default='./figs')
    parser.add_argument('--levels', nargs='+', default=['sfc', '850', '500', '200'])
    parser.add_argument('--times', nargs='+', default=None,
                        help="Valid times 'YYYYMMDDHH'; default all.")
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))
