import os

from supervision.assets import VideoAssets, download_assets


os.makedirs("data", exist_ok=True)
os.chdir("data")
download_assets(VideoAssets.VEHICLES)