# 这是一个示例 Python 脚本。


# 加载预训练模型
from ultralytics import YOLO
import cv2
# 加载预训练模型
model = YOLO("yolov8n.pt")  # 使用YOLOv8 Nano版本作为起点
# 查看版本信息
print(model.info())
print("加载完成")
# model('E:/dogcat.jpg', save=False, show=True) 窗口一闪而过
#解决
result=model('E:/61.png', show=False)
image=result[0].plot()
cv2.imshow('yolo result',image)
cv2.waitKey(0) #按任意键关闭窗口
cv2.destroyAllWindows()