from roboflow import Roboflow
import os

def download():
    rf = Roboflow(api_key="ouoUjVG1pG8SB5fwDwY7")
    project = rf.workspace("fish-pvnao").project("heavyrain-j1i5v")
    version = project.version(1)
    dataset = version.download("coco")
    print(f"Dataset downloaded to: {dataset.location}")

if __name__ == "__main__":
    download()
