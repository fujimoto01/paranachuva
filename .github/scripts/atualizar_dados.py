"""Atualiza os dados do site (pasta data/) com o que a NOAA publicou desde a última execução.

Chuva diária: CPC Global Unified Gauge-Based Daily Precipitation (NOAA PSL), 0,5°.
SPI: SPI-Gamma da NOAA NCEI/NIDIS sobre o GPCC 1,0°.

O estado é a própria pasta data/: baixa de novo só o último ano do manifest (se incompleto) até o
ano corrente, regrava os data/precip/AAAA.bin desses anos, atualiza o manifest e refaz o spi.bin.
A grade não muda, então regions.json e geo/ ficam como estão.

    python .github/scripts/atualizar_dados.py [--data-dir data] [--skip-spi]
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import pathlib
import subprocess
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

CPC_URL = "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc"
SPI_URL = "https://www.ncei.noaa.gov/pub/data/nidis/gpcc/spi-gamma/gpcc-spi-gamma-{w:02d}.nc"
SPI_START = "1979-01"
SPI_CLIP = 3.5


def download(url: str, target: pathlib.Path) -> pathlib.Path:
    """Baixa com curl. A velocidade da NOAA varia por conexão: abaixo de 100 kB/s por 20 s o curl
    desiste e a nova tentativa retoma (-C -) em outra conexão. -f recusa resposta de erro."""
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-fsSL", "--retry", "20", "--retry-all-errors", "--retry-delay", "2", "-C", "-",
           "--connect-timeout", "30", "--speed-limit", "100000", "--speed-time", "20", "-o", str(target), url]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Falha ao baixar {url}: curl saiu com {done.returncode} {done.stderr.strip()}")
    return target


def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def years_to_refresh(entries: list[dict], today: dt.date) -> list[int]:
    """Do último ano publicado (ou do seguinte, se ele já fechou) até o ano corrente."""
    last = entries[-1]
    first = last["year"] if last["days"] < days_in_year(last["year"]) else last["year"] + 1
    return list(range(first, today.year + 1))


def load_year(nc: pathlib.Path, year: int, lats: list[float], lons: list[float]) -> xr.DataArray:
    """Chuva diária do ano na grade publicada: lon em -180..180, lat e lon crescentes, só dias do
    ano, sem duplicatas e contíguos a partir de 1º de janeiro."""
    with xr.open_dataset(nc, decode_cf=True, mask_and_scale=True) as ds:
        lon = ds.lon.values
        ds = ds.assign_coords(lon=np.where(lon > 180.0, lon - 360.0, lon)).sortby(["lat", "lon"])
        precip = ds["precip"].sel(lat=slice(lats[0] - 0.01, lats[-1] + 0.01), lon=slice(lons[0] - 0.01, lons[-1] + 0.01)).load()
    if [round(float(v), 4) for v in precip.lat.values] != lats or [round(float(v), 4) for v in precip.lon.values] != lons:
        raise ValueError(f"{year}: grade do download diverge da grade publicada")

    times = pd.DatetimeIndex(precip.time.values)
    precip = precip.isel(time=np.flatnonzero(times.year == year))
    _, first_idx = np.unique(precip.time.values, return_index=True)
    precip = precip.isel(time=np.sort(first_idx))
    n = precip.sizes["time"]
    if n == 0:
        raise ValueError(f"{year}: nenhum dia do ano no arquivo")
    if not (precip.time.values == pd.date_range(f"{year}-01-01", periods=n, freq="D").values).all():
        raise ValueError(f"{year}: dias não contíguos a partir de 1º de janeiro")
    return precip.transpose("time", "lat", "lon")


def encode(values: np.ndarray, scale: float, nodata: int, lo: float, hi: float) -> np.ndarray:
    """Float (NaN = ausente) -> Int16 little-endian escalado. Ausente nunca vira 0."""
    out = np.full(values.shape, nodata, dtype="<i2")
    ok = np.isfinite(values)
    out[ok] = np.rint(np.clip(values[ok], lo, hi) * scale).astype("<i2")
    return out


def update_precip(data: pathlib.Path, tmp: pathlib.Path, today: dt.date) -> None:
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries, grid = manifest["years"], manifest["grid"]
    scale, nodata = manifest["scale"], manifest["nodata"]

    for year in years_to_refresh(entries, today):
        print(f"==> CPC {year}", flush=True)
        old = next((e for e in entries if e["year"] == year), None)
        try:
            nc = download(CPC_URL.format(year=year), tmp / f"precip.{year}.nc")
            precip = load_year(nc, year, grid["lats"], grid["lons"])
        except (RuntimeError, ValueError) as exc:
            # Nos primeiros dias de janeiro a NOAA ainda não publicou o ano novo: não é erro.
            if old is None and year == today.year and today.month == 1 and today.day <= 10:
                print(f"    {year} ainda não disponível na NOAA ({exc}); fica para a próxima execução")
                break
            raise
        days = precip.sizes["time"]
        if old and days < old["days"]:
            raise ValueError(f"{year}: download com {days} dias, menos que os {old['days']} já publicados")

        encode(precip.values, scale, nodata, 0, 32767 / scale).tofile(data / "precip" / f"{year}.bin")
        dates = [str(t)[:10] for t in precip.time.values]
        entry = {"year": year, "file": f"precip/{year}.bin", "start": dates[0], "end": dates[-1], "days": days}
        entries = sorted([e for e in entries if e["year"] != year] + [entry], key=lambda e: e["year"])
        print(f"    {dates[0]} a {dates[-1]} ({days} dias)")
        if days < days_in_year(year):
            break  # só o último ano da série pode estar incompleto

    manifest["years"] = entries
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")


def update_spi(data: pathlib.Path, tmp: pathlib.Path) -> None:
    """Recorta o SPI da NOAA para o PR e leva cada célula de 1,0° do GPCC às células de 0,5° do CPC
    que ela contém. Nada é recalculado."""
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = manifest["spi"]
    lats, lons = manifest["grid"]["lats"], manifest["grid"]["lons"]
    glat = xr.DataArray([math.floor(v) + 0.5 for v in lats], dims="y")
    glon = xr.DataArray([(math.floor(v) + 0.5) % 360.0 for v in lons], dims="x")

    arrays, month_lists, attrs = [], [], {}
    for w in meta["windows"]:
        print(f"==> SPI-{w:02d}", flush=True)
        nc = download(SPI_URL.format(w=w), tmp / f"spi-{w:02d}.nc")
        with xr.open_dataset(nc) as ds:
            da = ds[f"spi_{w:02d}"].sel(time=slice(f"{SPI_START}-01", None))
            month_lists.append([str(t)[:7] for t in da.time.values])
            arrays.append(np.asarray(da.sel(lat=glat, lon=glon).transpose("time", "y", "x").values, dtype="float32"))
            attrs = attrs or dict(da.attrs)
        nc.unlink()

    common = sorted(set.intersection(*(set(m) for m in month_lists)))
    for a, b in zip(common, common[1:]):
        if (int(b[:4]) * 12 + int(b[5:])) - (int(a[:4]) * 12 + int(a[5:])) != 1:
            raise ValueError(f"SPI com meses faltando entre {a} e {b}")
    if common[0] != meta["start"] or len(common) < meta["months"]:
        raise ValueError(f"SPI baixado ({common[0]}, {len(common)} meses) cobre menos que o publicado ({meta['start']}, {meta['months']})")

    stacked = np.stack([arr[[m.index(c) for c in common]] for arr, m in zip(arrays, month_lists)], axis=1)
    encode(stacked, meta["scale"], manifest["nodata"], -SPI_CLIP, SPI_CLIP).tofile(data / "spi.bin")
    print(f"    {common[0]} + {len(common)} meses (antes: {meta['months']})")
    meta["months"] = len(common)
    if "calibration_start_year" in attrs and "calibration_end_year" in attrs:
        meta["calibration"] = f"{int(attrs['calibration_start_year'])}-{int(attrs['calibration_end_year'])}"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Atualiza data/ com a chuva diária do CPC e o SPI da NOAA.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--skip-spi", action="store_true")
    args = ap.parse_args()
    data_dir = pathlib.Path(args.data_dir)
    with tempfile.TemporaryDirectory() as t:
        update_precip(data_dir, pathlib.Path(t), dt.date.today())
        if not args.skip_spi:
            try:
                update_spi(data_dir, pathlib.Path(t))
            except RuntimeError as exc:
                # Falha de rede na NCEI não segura a chuva diária: o spi.bin anterior continua valendo.
                print(f"::warning::SPI não atualizado ({exc}); mantido o arquivo anterior")
