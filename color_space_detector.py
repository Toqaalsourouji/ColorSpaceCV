import os, glob
import cv2
import numpy as np



#Here I am telling opencv to convert every color space we support to BGR 
TO_BGR = {
    'RGB'  : cv2.COLOR_RGB2BGR,
    'BGR'  : None,
    'HSV'  : cv2.COLOR_HSV2BGR,
    'HLS'  : cv2.COLOR_HLS2BGR,
    'LAB'  : cv2.COLOR_LAB2BGR,
    'YUV'  : cv2.COLOR_YUV2BGR,
    # 'YCrCb': cv2.COLOR_YCrCb2BGR,
    'XYZ'  : cv2.COLOR_XYZ2BGR,
    'GRAY' : cv2.COLOR_GRAY2BGR,
    'RGBA' : cv2.COLOR_RGBA2BGR,
    'BGRA' : cv2.COLOR_BGRA2BGR,
}

#created a list of the all color spaces out tool supports 
supportedCS = list(TO_BGR.keys())

#mapping to fix folder name mismatches
FOLDER_MAP = {'YCRCB':'YCrCb'}


#This function is the logic of out detector:
#1) We identify the folder of the image which is the true space
#2) next we load the raw array .npy image
#3) if the channels are 1 or 4 we return GRAY or RGBA or BGRA
#4) if the channels are 3 for each color space we converted to BGR we compute the mean square error
#against the reference image JPG and we pick the color space with the lowest MSE

def detect_color_space_by_mse(npy_path):

    space_folder = os.path.basename(os.path.dirname(npy_path))        
    space_key    = FOLDER_MAP.get(space_folder, space_folder)          

    
    arr = np.load(npy_path)
    shape = arr.shape


    # GRAY: single‐channel or single‐channel with dummy dimension
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return 'GRAY'


    # if the array has 4 channels, it's RGBA or BGRA 
    if arr.ndim == 3 and shape[2] == 4:
        return space_key


    #for 3-channel images, try to detect the color space by comparing to the reference JPG
    # 1) Try the standard folder layout first
    jpg_name = os.path.splitext(os.path.basename(npy_path))[0] + ".jpg" 
    # jpg_path = os.path.join("converted_images", space_folder, jpg_name)
    # true_bgr = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
    # 2) Fallback: maybe the .jpg lives right next to the .npy
    # if true_bgr is None:
    fallback = os.path.join(os.path.dirname(npy_path), jpg_name)
    true_bgr = cv2.imread(fallback, cv2.IMREAD_UNCHANGED)
    if true_bgr is None:
        raise FileNotFoundError(
            f"Cannot find reference JPG in either:\n"
            # f"  {jpg_path}\n"
            f"  {fallback}"
        )

    #ensure the JPG image is 3-channel BGR
    if true_bgr.ndim == 2:
        true_bgr = cv2.cvtColor(true_bgr, cv2.COLOR_GRAY2BGR)
    elif true_bgr.shape[2] > 3:
        true_bgr = true_bgr[..., :3]  

    #initialize variables to track the best matching color space and lowest error
    best_space, best_err = None, float('inf')
    #try each supported 3-channel color space
    for space in supportedCS:
        flag = TO_BGR[space]
        try:
           
            recon = arr if flag is None else cv2.cvtColor(arr, flag)
        except:
            continue  

        
        if recon.ndim == 3 and recon.shape[2] > 3:
            recon = recon[..., :3]
        if recon.ndim == 2:
            recon = cv2.cvtColor(recon, cv2.COLOR_GRAY2BGR)

        
        if recon.shape[:2] != true_bgr.shape[:2]:
            recon = cv2.resize(recon, (true_bgr.shape[1], true_bgr.shape[0]))

        #Compute mean squared error between reconstructed and reference images
        err = np.mean((recon.astype(np.float32) - true_bgr.astype(np.float32))**2)
        if err < best_err:
            best_err, best_space = err, space  

    return best_space  

if __name__ == "__main__":
    # Main script: iterate through all folders in 'converted_images_npy'
    for folder in os.listdir("converted_images_npy"):
        folder_path = os.path.join("converted_images_npy", folder)
        if not os.path.isdir(folder_path):
            continue  

        #get all .npy files in the folder
        files = glob.glob(os.path.join(folder_path, "*.npy"))
        if not files:
            print(f"[!] No .npy in {folder}")
            continue

        #for testing i am using the first .npy in each folder 
        sample = files[0]
        detected = detect_color_space_by_mse(sample)
        print(f"Sample from {folder}: {os.path.basename(sample)} → {detected}")
