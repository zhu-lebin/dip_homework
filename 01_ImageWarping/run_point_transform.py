import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    # print(image.shape)
    # print(source_pts)
    warped_image = np.array(image)
    mask_warp = np.zeros(image.shape[:2], dtype=np.uint8)
    # 
    ### FILL: 基于MLS or RBF 实现 image warping
    warped_image = np.zeros(image.shape, dtype=np.uint8)
    n = len(source_pts)
    A = np.zeros((n+3,n+3))
    b = np.zeros((n+3, 2))
    b[0:n,:] = target_pts
    b[-3:,:] = 0
    A[0:n,0:n] = np.array([[base_func(source_pts[i],source_pts[j]) for j in range(n)] for i in range(n)])
    A[-1:,0:n] = 1
    A[n:n+2,0:n] = source_pts.T
    A[0:n,n:n+2] = source_pts
    A[0:n,-1:] = 1
    x = np.linalg.solve(A, b)
    affine_matrix = np.array([[x[n,0], x[n,1]], [x[n+1,0], x[n+1,1]]]).T
    #由于image的坐标相对于gradio给的控制点坐标是先y后x，所以这里要调换一下xy，做运算之后再调换回来
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            p = np.array([j,i])
            q = np.zeros(2)
            for k in range(n):
                q += x[k,0:2].T * base_func(p, source_pts[k])
            q += np.dot(affine_matrix, p)
            q += x[-1,0:2].T
            q = q.astype(int)
            if q[0] >= 0 and q[0] < image.shape[1] and q[1] >= 0 and q[1] < image.shape[0]:
                warped_image[q[1],q[0]] = image[i,j] 
                mask_warp[q[1],q[0]] = 255
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask_warp[i,j] == 0:
                points_near = []
                for s in range(i-2,i+2):
                    for t in range(j-2,j+2):
                        if s >= 0 and s < image.shape[0] and t >= 0 and t < image.shape[1] and mask_warp[s,t] == 255:
                            points_near.append([t,s])  
                if len(points_near) > 0:
                    warped_image[i,j] = np.mean([warped_image[pt[1],pt[0]] for pt in points_near])
                    mask_warp[i,j] = 255       
    return warped_image

def base_func(p,q):
    # r = np.linalg.norm(p-q)
    d = 10
    miu = 1
    return np.power(((p[0]-q[0])*(p[0]-q[0]) + (p[1]-q[1])*(p[1]-q[1]) + d*d),miu/2)


def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
