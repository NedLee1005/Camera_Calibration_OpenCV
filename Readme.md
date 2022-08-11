# Camera calibration 
it's welcome to fork from  this directory

## 執行 Calibration.py 
```bash
$python .\calibration.py 
# or
$python .\calibration.py -s  <row> <col> -m <正方形長度> -S <儲存位置> -l <供矯正的圖片位置>
```


##  command option
```
optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  Enter number of row/col points ex:6 7
  -m MM, --mm MM        enter square length of chessboard defaut is 1  
  -S SAVE_DIR, --save_dir SAVE_DIR
                        directory to save undistorted images
  -l LOAD_DIR, --load_dir LOAD_DIR
                        directory that contain images for camera calibration
```
## 執行完畢
### 會有三個產出，分別為
* output undistorted images (location at SAVE_DIR => default: *CalibrateImage*)
* cam_distortion.npy (distortion coefficients)
  * ![](https://i.imgur.com/KphYWM1.png)

* cam_matrix.npy (camera matrix)
  * ![](https://i.imgur.com/TnojboV.png)

