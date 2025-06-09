#stream
#使用 stream=True 用于处理长视频或大型数据集，以有效管理内存。
# 当 stream=False在这种情况下，所有帧或数据点的结果都会存储在内存中，这可能会迅速累加，
# 并导致大量输入出现内存不足错误。
# 与此形成鲜明对比的是 stream=True 利用生成器，只将当前帧或数据点的结果保存在内存中，
# 从而大大减少了内存消耗，并防止出现内存不足的问题。

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["E:/61.png", "E:/dogcat.jpg"],stream=True)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    #result.save(filename="result.jpg")  # save to disk