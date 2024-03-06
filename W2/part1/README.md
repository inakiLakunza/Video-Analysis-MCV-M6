
# Use Case part 1

To execute this part you create a conda environment with python 3.9.6

```bash
conda create --name detectron python==3.9.6 & conda activate detectron
```
Install the requirements by 
```bash
pip install -r requirements.txt

```

To create the data folder with the frames in order to use detectron you first need to create the frames from the video.

```python
python create_frame_dataset.py
```

Go inside the file and change the path where you have your sequence

```python
   #Concrete case
    video_path = '../../AICity_data/train/S03/c010/vdo.avi'
```

After go to the file AIcityDataset.py and change the paths to fit with your xml annotation path.

```python
    python AIcityDataset.py
```


## Experiments

To evaluate your data you have the **Evaluator.ipynb** with the steps defined to do inference

If you want to finetunne your data you have the **Finetune.ipynb** file with the steps to do it step by step


## Custom Loaders

If you have your own annotations or you want to create your own partition you need to add the functionality in the file **data_loaders.py**



