import os
import sys
import webbrowser
import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_ALL # type: ignore
from typing import Any, Callable, Tuple, Optional
import cv2
from PIL import Image, ImageOps
import glob
import datetime # 新增: 用于生成唯一文件名
import numpy as np # 新增: 用于图像格式转换

import roop.globals
import roop.metadata
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total # get_video_frame_total 可能不再需要
from roop.face_reference import (
    get_face_reference, # 可能不再需要
    set_face_reference, # 可能不再需要
    clear_face_reference, # 可能不再需要
)
from roop.predictor import predict_frame, clear_predictor # predict_frame 和 clear_predictor 可能不再与预览相关
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

# PREVIEW 相关的全局变量已被移除
# PREVIEW = None
# PREVIEW_MAX_HEIGHT = 700
# PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

# preview_label 和 preview_slider 已被移除
source_label = None
target_label = None
status_label = None
status_icon = None

realtime_label = None
realtime_running = False
realtime_cap = None

overlay_frame = None
overlay_text = None

# --- 新增录制相关全局变量 ---
recording_active = False
video_writer = None
output_video_path = None
record_button = None # 用于之后更新按钮文本
RECORD_WIDTH = 800 # 录制视频宽度
RECORD_HEIGHT = 600 # 录制视频高度
FPS_RECORDING = 8.0 # 录制视频帧率


# todo: remove by native support -> https://github.com/TomSchimansky/CustomTkinter/issues/934
class CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT
    # PREVIEW 的创建已被移除
    ROOT = create_root(start, destroy)
    return ROOT


def create_switch(root, text, var, relx, rely, command=None):
    """统一创建开关控件"""
    switch = ctk.CTkSwitch(
        root,
        text=text,
        variable=var,
        cursor="hand2",
        command=command,
        font=("Segoe UI", 12),
    )
    switch.place(relx=relx, rely=rely)
    return switch


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, realtime_label
    global status_icon, overlay_frame, overlay_text
    global record_button # 引用全局 record_button

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = CTk()
    root.minsize(1200, 800)
    root.title(f"Face Swap Studio - 高级换脸工具")
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # --- 左侧功能区 ---
    left_frame = ctk.CTkFrame(root, corner_radius=15)
    left_frame.place(relx=0.01, rely=0.01, relwidth=0.25, relheight=0.98)

    app_title = ctk.CTkLabel(
        left_frame, 
        text="Face Swap Studio", 
        font=("Segoe UI", 18, "bold"),
        anchor="center"
    )
    app_title.place(relx=0.5, rely=0.02, anchor="center")

    separator = ctk.CTkFrame(left_frame, height=2, fg_color=["#d0d0d0", "#45475a"])
    separator.place(relx=0.05, rely=0.05, relwidth=0.9)

    face_gallery_label = ctk.CTkLabel(
        left_frame, 
        text="人脸库", 
        anchor="w", 
        font=("Segoe UI", 14, "bold"),
        text_color=["#7289da", "#7867c5"]
    )
    face_gallery_label.place(relx=0.05, rely=0.07, relwidth=0.7, relheight=0.04)

    refresh_button = ctk.CTkButton(
        left_frame, 
        text="刷新", 
        cursor="hand2", 
        width=60,
        height=28,
        corner_radius=8,
        font=("Segoe UI", 12),
        command=lambda: load_face_gallery(face_gallery_frame)
    )
    refresh_button.place(relx=0.8, rely=0.07, relwidth=0.15)

    face_gallery_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color=["#f7f7f9", "#292a37"])
    face_gallery_frame.place(relx=0.05, rely=0.12, relwidth=0.9, relheight=0.16)
    
    load_face_gallery(face_gallery_frame)

    separator2 = ctk.CTkFrame(left_frame, height=2, fg_color=["#d0d0d0", "#45475a"])
    separator2.place(relx=0.05, rely=0.30, relwidth=0.9)

    source_title = ctk.CTkLabel(
        left_frame, 
        text="源图片", 
        anchor="w", 
        font=("Segoe UI", 14, "bold"),
        text_color=["#7289da", "#7867c5"]
    )
    source_title.place(relx=0.05, rely=0.32, relwidth=0.9, relheight=0.04)

    source_label = ctk.CTkLabel(
        left_frame,
        text="",
        fg_color=["#f7f7f9", "#292a37"],
        corner_radius=10,
        justify="center",
    )
    source_label.place(relx=0.05, rely=0.37, relwidth=0.9, relheight=0.15)

    source_button = ctk.CTkButton(
        left_frame, 
        text="选择源图片", 
        cursor="hand2", 
        height=32,
        corner_radius=8,
        font=("Segoe UI", 12),
        command=lambda: select_source_path()
    )
    source_button.place(relx=0.05, rely=0.53, relwidth=0.9, relheight=0.05)

    target_title = ctk.CTkLabel(
        left_frame, 
        text="目标图片/视频", 
        anchor="w", 
        font=("Segoe UI", 14, "bold"),
        text_color=["#7289da", "#7867c5"]
    )
    target_title.place(relx=0.05, rely=0.59, relwidth=0.9, relheight=0.04)

    target_label = ctk.CTkLabel(
        left_frame,
        text="",
        fg_color=["#f7f7f9", "#292a37"],
        corner_radius=10,
        justify="center",
    )
    target_label.place(relx=0.05, rely=0.64, relwidth=0.9, relheight=0.15)

    target_button = ctk.CTkButton(
        left_frame,
        text="选择目标文件",
        cursor="hand2",
        height=32,
        corner_radius=8,
        font=("Segoe UI", 12),
        command=lambda: select_target_path(),
    )
    target_button.place(relx=0.05, rely=0.80, relwidth=0.9, relheight=0.05)

    separator3 = ctk.CTkFrame(left_frame, height=2, fg_color=["#d0d0d0", "#45475a"])
    separator3.place(relx=0.05, rely=0.86, relwidth=0.9)

    roop.globals.keep_fps = True
    roop.globals.skip_audio = False
    roop.globals.keep_frames = False
    roop.globals.many_faces = False

    operations_frame = ctk.CTkFrame(
        left_frame,
        corner_radius=10,
        fg_color="transparent"
    )
    operations_frame.place(relx=0.05, rely=0.88, relwidth=0.9, relheight=0.1)

    start_button = ctk.CTkButton(
        operations_frame,
        text="开始处理",
        cursor="hand2",
        font=("Segoe UI", 13, "bold"),
        height=36,
        corner_radius=10,
        border_width=0,
        fg_color=["#7289da", "#7867c5"],
        hover_color=["#5d73bc", "#6a5aad"],
        command=lambda: select_output_path(start),
    )
    start_button.place(relx=0, rely=0, relwidth=0.32, relheight=1)
    
    # --- 修改：预览按钮替换为录制按钮 ---
    record_button = ctk.CTkButton(
        operations_frame,
        text="开始录制", # 初始文本
        cursor="hand2",
        font=("Segoe UI", 13, "bold"),
        height=36,
        corner_radius=10,
        border_width=0,
        # fg_color 和 hover_color 可以根据喜好设置，或与其他按钮保持一致
        command=lambda: toggle_recording() 
    )
    record_button.place(relx=0.34, rely=0, relwidth=0.32, relheight=1)
    
    realtime_button = ctk.CTkButton(
        operations_frame,
        text="实时换脸",
        cursor="hand2",
        font=("Segoe UI", 13, "bold"),
        height=36,
        corner_radius=10,
        border_width=0,
        command=lambda: toggle_real_time_face_swap(),
    )
    realtime_button.place(relx=0.68, rely=0, relwidth=0.32, relheight=1)

    # --- 右侧预览区 (现在主要用于实时换脸显示) ---
    right_frame = ctk.CTkFrame(root, corner_radius=15)
    right_frame.place(relx=0.27, rely=0.01, relwidth=0.72, relheight=0.98)
    
    overlay_frame = ctk.CTkFrame(
        right_frame, 
        fg_color=["#f7f7f9", "#292a37"],
        corner_radius=15
    )
    overlay_text = ctk.CTkLabel(
        overlay_frame,
        # --- 修改：更新提示文本 ---
        text="请点击'实时换脸'查看结果，或选择文件后点击'开始处理'",
        font=("Segoe UI", 16, "bold"),
        text_color=["#7289da", "#7867c5"]
    )
    overlay_frame.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.2, anchor="center") # 调整宽度以适应新文本
    overlay_text.place(relx=0.5, rely=0.5, anchor="center")
    
    realtime_label = ctk.CTkLabel(
        right_frame,
        text="",
        fg_color=["#f0f0f4", "#1e1e2e"],
        corner_radius=10
    )
    realtime_label.place(relx=0.025, rely=0.02, relwidth=0.95, relheight=0.9)

    status_frame = ctk.CTkFrame(
        right_frame,
        corner_radius=8,
        fg_color=["#f0f0f4", "#292a37"],
        height=30
    )
    status_frame.place(relx=0.025, rely=0.93, relwidth=0.95, relheight=0.05)
    
    status_icon = ctk.CTkLabel(
        status_frame,
        text="●", 
        font=("Segoe UI", 16),
        text_color=["#7289da", "#7867c5"],
        anchor="w"
    )
    status_icon.place(relx=0.01, rely=0.5, anchor="w")
    
    status_label = ctk.CTkLabel(
        status_frame,
        text="就绪 - 请选择源图片和目标文件",
        font=("Segoe UI", 12),
        justify="left",
        anchor="w"
    )
    status_label.place(relx=0.04, rely=0.5, anchor="w")

    return root

# --- 移除 create_preview, toggle_preview, init_preview, update_preview, update_face_reference, update_frame 函数 ---

def update_status(text: str) -> None:
    global status_icon, status_label, ROOT
    
    if "错误" in text or "失败" in text:
        status_icon.configure(text_color=["#e74c3c", "#e74c3c"])
    elif "成功" in text or "完成" in text or "保存" in text: # 添加 "保存"
        status_icon.configure(text_color=["#2ecc71", "#2ecc71"])
    elif "警告" in text:
        status_icon.configure(text_color=["#f39c12", "#f39c12"])
    elif "请" in text or "录制中" in text: # 添加 "录制中"
        status_icon.configure(text_color=["#3498db", "#3498db"])
    else:
        status_icon.configure(text_color=["#7289da", "#7867c5"])
    
    status_label.configure(text=text)
    if ROOT: # 确保 ROOT 存在
        ROOT.update()


def select_source_path(source_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_SOURCE, source_label

    if source_path is None:
        source_path = ctk.filedialog.askopenfilename(
            title="选择源图片", # 更新为中文
            initialdir=RECENT_DIRECTORY_SOURCE,
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.webp"),),
        )
    if is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        image = render_image_preview(source_path, (200, 200)) # 预览尺寸可以根据需要调整
        source_label.configure(image=image)
        if realtime_running: 
            update_status("源图片已更新，实时换脸将使用新图片")
    else:
        if source_path: # 如果用户取消选择，source_path 会是空字符串或 None
             update_status("错误 - 请选择有效的源图片文件")
        # roop.globals.source_path = None # 保持之前的状态或清除，取决于逻辑
        # source_label.configure(image=None)


def select_target_path(target_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_TARGET, target_label
    # PREVIEW 相关的 clear_face_reference() 和 PREVIEW.withdraw() 已移除

    if target_path is None:
        target_path = ctk.filedialog.askopenfilename(
            title="选择目标图片或视频", # 更新为中文
            initialdir=RECENT_DIRECTORY_TARGET,
            filetypes=(
                ("All files", "*.*"), # 允许选择所有文件，由 is_image/is_video 判断
                ("Image files", "*.jpg;*.jpeg;*.png;*.webp"),
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
            )
        )
    if is_image(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        image = render_image_preview(roop.globals.target_path, (200, 200)) # 预览尺寸可以根据需要调整
        target_label.configure(image=image)
        update_status("目标图片已选择")
    elif is_video(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        video_frame = render_video_preview(target_path, (200,200)) # 预览尺寸可以根据需要调整
        target_label.configure(image=video_frame)
        update_status("目标视频已选择")
    else:
        if target_path: # 如果用户取消选择，target_path 会是空字符串或 None
            update_status("错误 - 请选择有效的图片或视频文件作为目标")
        # roop.globals.target_path = None # 保持之前的状态或清除
        # target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT

    if not roop.globals.source_path:
        update_status("请先选择源图片")
        return
    if not roop.globals.target_path:
        update_status("请先选择目标文件")
        return

    if is_image(roop.globals.target_path):
        output_path_selected = ctk.filedialog.asksaveasfilename( # 重命名变量以避免与全局冲突
            title="保存图片输出文件",
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
            filetypes=(("PNG images", "*.png"), ("JPEG images", "*.jpg")),
        )
    elif is_video(roop.globals.target_path):
        output_path_selected = ctk.filedialog.asksaveasfilename(
            title="保存视频输出文件",
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
            filetypes=(("MP4 video", "*.mp4"), ("AVI video", "*.avi")),
        )
    else:
        update_status("错误 - 目标文件类型不支持处理")
        return # 直接返回，因为 output_path_selected 可能未定义

    if output_path_selected: # 检查用户是否选择了路径
        roop.globals.output_path = output_path_selected
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(roop.globals.output_path)
        start() # 调用核心处理函数
    else:
        update_status("输出操作已取消")


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    image = ImageOps.contain(image, size, Image.LANCZOS) 
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[ctk.CTkImage]: # 返回 Optional
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        update_status(f"错误 - 无法打开视频文件: {video_path}")
        return None
        
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    capture.release() # 及时释放
    # cv2.destroyAllWindows() # 通常在主循环结束时调用，或不需要在此处调用
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    else:
        update_status(f"错误 - 无法从视频读取帧: {video_path}")
        return None

# --- 新增录制控制函数 ---
def toggle_recording():
    global recording_active, record_button

    if not realtime_running:
        update_status("请先开启实时换脸功能才能录制")
        return

    if not recording_active:
        start_recording()
    else:
        stop_recording()

    if record_button: # 更新按钮文本
        record_button.configure(text="停止录制" if recording_active else "开始录制")

def start_recording():
    global recording_active, video_writer, output_video_path, realtime_cap

    if not realtime_cap or not realtime_cap.isOpened():
        update_status("错误 - 摄像头未准备好，无法开始录制")
        return

    filename = f"realtime_record_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_video_path = os.path.join(os.getcwd(), filename) # 保存在当前目录

    # 使用固定的录制尺寸和FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for .mp4
    try:
        video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS_RECORDING, (RECORD_WIDTH, RECORD_HEIGHT))
        if not video_writer.isOpened():
            raise IOError("cv2.VideoWriter failed to open.")
        recording_active = True
        update_status(f"录制已开始，保存至: {os.path.basename(output_video_path)}")
    except Exception as e:
        video_writer = None
        recording_active = False
        output_video_path = None
        update_status(f"错误 - 录制器初始化失败: {e}")
        if record_button:
            record_button.configure(text="开始录制")


def stop_recording():
    global recording_active, video_writer, output_video_path

    recording_active = False
    if video_writer:
        video_writer.release()
        video_writer = None
        update_status(f"录制已停止，视频保存至: {output_video_path}")
        # output_video_path = None # 保留路径以便用户查看
    else:
        # 如果 video_writer 为 None 但 stop_recording 被调用 (例如实时换脸停止时)
        update_status("录制已停止") 
        
    if record_button: # 总是重置按钮文本
         record_button.configure(text="开始录制")


def toggle_real_time_face_swap():
    global realtime_running
    if not realtime_running:
        start_real_time_face_swap()
    else:
        stop_real_time_face_swap()


def start_real_time_face_swap():
    global realtime_running, realtime_cap, overlay_frame

    if not roop.globals.source_path:
        update_status("请先选择源人脸图片")
        return

    try:
        source_image = cv2.imread(roop.globals.source_path)
        if source_image is None:
            update_status(f"错误 - 无法读取源图片: {roop.globals.source_path}")
            return
        source_face = get_one_face(source_image)
    except Exception as e:
        update_status(f"读取源图片或分析人脸时出错: {e}")
        return
        
    if not source_face:
        update_status("错误 - 源图片未检测到人脸")
        return

    realtime_cap = cv2.VideoCapture(0) # 默认摄像头
    if not realtime_cap.isOpened():
        update_status("错误 - 无法打开摄像头")
        realtime_cap = None # 确保cap被重置
        return

    if overlay_frame:
        overlay_frame.place_forget()
        
    realtime_running = True
    update_status("实时换脸中...")
    if record_button: # 确保录制按钮是“开始录制”状态
        record_button.configure(text="开始录制")
    update_realtime_frame(source_face)


def stop_real_time_face_swap():
    global realtime_running, realtime_cap, overlay_frame, overlay_text, recording_active

    if recording_active: # 如果正在录制，先停止录制
        stop_recording()
        
    realtime_running = False
    if realtime_cap:
        realtime_cap.release()
        realtime_cap = None
    
    realtime_label.configure(image=None, text="")
    
    if overlay_frame and overlay_text:
        overlay_frame.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.2, anchor="center")
        overlay_text.place(relx=0.5, rely=0.5, anchor="center")
    
    update_status("实时换脸已停止")
    if record_button: # 确保录制按钮也重置
        record_button.configure(text="开始录制")


def update_realtime_frame(source_face_initial):
    global realtime_running, realtime_cap, video_writer, recording_active, source_label

    if not realtime_running or not realtime_cap or not realtime_cap.isOpened():
        if realtime_running: # 如果仍在运行但摄像头出错了
            update_status("错误 - 摄像头连接中断")
            stop_real_time_face_swap()
        return

    # 动态获取最新的源人脸 (如果用户在实时换脸过程中更改了源图片)
    current_source_face = source_face_initial
    if roop.globals.source_path:
        try:
            new_source_img = cv2.imread(roop.globals.source_path)
            if new_source_img is not None:
                potential_new_face = get_one_face(new_source_img)
                if potential_new_face is not None:
                    current_source_face = potential_new_face
                    # (可选) 可以在状态栏提示源人脸已更新，但避免过于频繁更新状态栏扰乱录制信息
                    # current_image_name = os.path.basename(roop.globals.source_path)
                    # status_label.configure(text=f"当前使用: {current_image_name}")
        except Exception as e:
            print(f"更新源人脸时出错: {e}") # 记录到控制台，不干扰UI

    ret, frame = realtime_cap.read()
    if not ret:
        update_status("摄像头帧获取失败，实时换脸停止")
        stop_real_time_face_swap() # 这会处理录制停止
        return

    frame = cv2.flip(frame, 1) # 镜像翻转，使预览更自然

    target_face = get_one_face(frame) # 从摄像头帧中获取人脸
    processed_frame = frame.copy() # 复制原始帧以进行处理

    if target_face and current_source_face: # 确保源人脸和目标人脸都存在
        try:
            for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                processed_frame = frame_processor.process_frame(current_source_face, target_face, processed_frame)
        except Exception as e:
            print(f"处理帧时出错: {e}") # 打印错误但继续，避免中断

    # --- 准备用于显示和录制的图像 ---
    pil_image_rgb = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    
    # 适应到 RECORD_WIDTH x RECORD_HEIGHT，保持纵横比，多余部分用黑色填充
    contained_image = ImageOps.contain(pil_image_rgb, (RECORD_WIDTH, RECORD_HEIGHT), Image.LANCZOS)
    
    final_image_for_display_and_video = Image.new('RGB', (RECORD_WIDTH, RECORD_HEIGHT), (0, 0, 0)) # 黑色背景
    paste_x = (RECORD_WIDTH - contained_image.width) // 2
    paste_y = (RECORD_HEIGHT - contained_image.height) // 2
    final_image_for_display_and_video.paste(contained_image, (paste_x, paste_y))

    # 显示到UI
    tk_image = ctk.CTkImage(final_image_for_display_and_video, size=(RECORD_WIDTH, RECORD_HEIGHT))
    realtime_label.configure(image=tk_image)

    # 录制帧
    if recording_active and video_writer and video_writer.isOpened():
        try:
            # 将Pillow Image (RGB) 转换为 OpenCV frame (BGR)
            frame_to_write_bgr = cv2.cvtColor(np.array(final_image_for_display_and_video), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_to_write_bgr)
        except Exception as e:
            update_status(f"录制帧写入错误: {e}")
            stop_recording() # 发生错误时停止录制
            if record_button: record_button.configure(text="开始录制")


    if realtime_running: # 确保在停止后不再次调用
        realtime_label.after(30, lambda: update_realtime_frame(current_source_face)) # ~33 FPS，可以调整


def load_face_gallery(gallery_frame):
    """加载人脸库图片"""
    for widget in gallery_frame.winfo_children():
        widget.destroy()
    
    faces_dir = "faces"
    if not os.path.exists(faces_dir):
        try:
            os.makedirs(faces_dir)
            # (可选) 可以添加一条消息提示用户 faces 文件夹已创建
        except OSError as e:
            update_status(f"错误 - 无法创建 faces 文件夹: {e}")
            no_dir_label = ctk.CTkLabel(gallery_frame, text=f"错误: 无法创建 '{faces_dir}' 文件夹")
            no_dir_label.pack(expand=True, fill="both", padx=10, pady=10)
            return

    face_images_patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    face_images = []
    for pattern in face_images_patterns:
        face_images.extend(glob.glob(os.path.join(faces_dir, pattern)))
    
    if not face_images:
        no_image_label = ctk.CTkLabel(gallery_frame, text="人脸库为空，请添加图片到 'faces' 文件夹")
        no_image_label.pack(expand=True, fill="both", padx=10, pady=10)
        return
    
    max_per_row = gallery_frame.winfo_width() // 60 if gallery_frame.winfo_width() > 60 else 4 # 动态计算或固定
    if max_per_row == 0: max_per_row = 4 # 确保不为0
    thumb_size = (48, 48)

    for idx, face_path in enumerate(face_images):
        try:
            img = Image.open(face_path)
            # 使用 ImageOps.fit 来裁剪和缩放以填充缩略图区域，避免变形
            img_thumb = ImageOps.fit(img, thumb_size, Image.LANCZOS)
            tk_img = ctk.CTkImage(img_thumb, size=thumb_size)
            
            btn = ctk.CTkButton(
                gallery_frame,
                image=tk_img,
                text="",
                width=thumb_size[0],
                height=thumb_size[1],
                fg_color="transparent", # 使按钮背景透明，只显示图片
                hover_color=["#e0e0e0", "#3a3a3a"], # 添加悬停效果
                command=lambda p=face_path: select_source_path(p),
            )
            # 使用 pack 代替 grid 以便更好地处理动态数量的图片和自动换行
            btn.pack(side=ctk.LEFT, padx=4, pady=4) 
            # 如果需要严格的网格布局，需要提前计算好行和列
            # btn.grid(row=idx // max_per_row, column=idx % max_per_row, padx=6, pady=4)

        except Exception as e:
            print(f"加载人脸库图片错误: {face_path} - {str(e)}")
            error_label = ctk.CTkLabel(gallery_frame, text=f"无法加载\n{os.path.basename(face_path)}", font=("Segoe UI", 8))
            error_label.pack(side=ctk.LEFT, padx=4, pady=4, ipadx=2, ipady=2)


# 主程序入口示例 (如果这是主文件)
if __name__ == '__main__':
    # 定义一个简单的 start 函数占位符，实际项目中它会执行核心处理
    def placeholder_start_processing():
        if roop.globals.source_path and roop.globals.target_path and roop.globals.output_path:
            update_status(f"开始处理: {roop.globals.source_path} -> {roop.globals.target_path} 保存至 {roop.globals.output_path}")
            # 在这里调用你的核心换脸处理逻辑
            # roop.core.start() # 假设 roop.core.start() 是处理函数
            # 模拟处理完成
            import time; time.sleep(3) 
            update_status(f"处理完成! 文件已保存: {roop.globals.output_path}")
        else:
            update_status("错误 - 处理所需信息不完整。")

    # 定义销毁函数
    def on_destroy():
        global realtime_running, recording_active
        if recording_active:
            stop_recording() # 确保退出前保存录制
        if realtime_running:
            stop_real_time_face_swap() # 停止摄像头等
        
        if realtime_cap:
            realtime_cap.release()
        cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口
        
        if ROOT:
            ROOT.quit()
            ROOT.destroy()
        sys.exit()

    main_window = init(placeholder_start_processing, on_destroy)
    main_window.mainloop()