import os
import sys
import webbrowser
import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_ALL
from typing import Any, Callable, Tuple, Optional
import cv2
from PIL import Image, ImageOps
import glob

import roop.globals
import roop.metadata
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total
from roop.face_reference import (
    get_face_reference,
    set_face_reference,
    clear_face_reference,
)
from roop.predictor import predict_frame, clear_predictor
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
status_icon = None

realtime_label = None
realtime_running = False
realtime_cap = None

overlay_frame = None
overlay_text = None


# todo: remove by native support -> https://github.com/TomSchimansky/CustomTkinter/issues/934
class CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

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

    # 应用标题
    app_title = ctk.CTkLabel(
        left_frame, 
        text="Face Swap Studio", 
        font=("Segoe UI", 18, "bold"),
        anchor="center"
    )
    app_title.place(relx=0.5, rely=0.02, anchor="center")

    # 分隔线
    separator = ctk.CTkFrame(left_frame, height=2, fg_color=["#d0d0d0", "#45475a"])
    separator.place(relx=0.05, rely=0.05, relwidth=0.9)

    # 人脸库标题
    face_gallery_label = ctk.CTkLabel(
        left_frame, 
        text="人脸库", 
        anchor="w", 
        font=("Segoe UI", 14, "bold"),
        text_color=["#7289da", "#7867c5"]
    )
    face_gallery_label.place(relx=0.05, rely=0.07, relwidth=0.7, relheight=0.04)

    # 刷新按钮
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

    # 人脸库区域（自动多行显示，支持多种格式）
    face_gallery_frame = ctk.CTkFrame(left_frame, corner_radius=10, fg_color=["#f7f7f9", "#292a37"])
    face_gallery_frame.place(relx=0.05, rely=0.12, relwidth=0.9, relheight=0.16)
    
    # 加载人脸库图片
    load_face_gallery(face_gallery_frame)

    # 分隔线
    separator2 = ctk.CTkFrame(left_frame, height=2, fg_color=["#d0d0d0", "#45475a"])
    separator2.place(relx=0.05, rely=0.30, relwidth=0.9)

    # 源图片标题
    source_title = ctk.CTkLabel(
        left_frame, 
        text="源图片", 
        anchor="w", 
        font=("Segoe UI", 14, "bold"),
        text_color=["#7289da", "#7867c5"]
    )
    source_title.place(relx=0.05, rely=0.32, relwidth=0.9, relheight=0.04)

    # 源图片显示区
    source_label = ctk.CTkLabel(
        left_frame,
        text="",
        fg_color=["#f7f7f9", "#292a37"],
        corner_radius=10,
        justify="center",
    )
    source_label.place(relx=0.05, rely=0.37, relwidth=0.9, relheight=0.15)

    # 选择源按钮
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

    # 目标图片/视频标题
    target_title = ctk.CTkLabel(
        left_frame, 
        text="目标图片/视频", 
        anchor="w", 
        font=("Segoe UI", 14, "bold"),
        text_color=["#7289da", "#7867c5"]
    )
    target_title.place(relx=0.05, rely=0.59, relwidth=0.9, relheight=0.04)

    # 目标图片/视频显示区
    target_label = ctk.CTkLabel(
        left_frame,
        text="",
        fg_color=["#f7f7f9", "#292a37"],
        corner_radius=10,
        justify="center",
    )
    target_label.place(relx=0.05, rely=0.64, relwidth=0.9, relheight=0.15)

    # 选择目标按钮
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

    # 分隔线
    separator3 = ctk.CTkFrame(left_frame, height=2, fg_color=["#d0d0d0", "#45475a"])
    separator3.place(relx=0.05, rely=0.86, relwidth=0.9)

    # 设置全局默认值
    roop.globals.keep_fps = True
    roop.globals.skip_audio = False
    roop.globals.keep_frames = False
    roop.globals.many_faces = False

    # 操作按钮区 - 使用更现代的样式
    operations_frame = ctk.CTkFrame(
        left_frame,
        corner_radius=10,
        fg_color="transparent"
    )
    operations_frame.place(relx=0.05, rely=0.88, relwidth=0.9, relheight=0.1)

    # 使用更美观的按钮布局和设计
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
    
    preview_button = ctk.CTkButton(
        operations_frame,
        text="预览效果",
        cursor="hand2",
        font=("Segoe UI", 13, "bold"),
        height=36,
        corner_radius=10,
        border_width=0,
        command=lambda: toggle_preview()
    )
    preview_button.place(relx=0.34, rely=0, relwidth=0.32, relheight=1)
    
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

    # --- 右侧预览区 ---
    right_frame = ctk.CTkFrame(root, corner_radius=15)
    right_frame.place(relx=0.27, rely=0.01, relwidth=0.72, relheight=0.98)
    
    # 创建半透明的提示覆盖层
    overlay_frame = ctk.CTkFrame(
        right_frame, 
        fg_color=["#f7f7f9", "#292a37"],
        corner_radius=15
    )
    overlay_text = ctk.CTkLabel(
        overlay_frame,
        text="请点击'预览效果'或'实时换脸'查看结果",
        font=("Segoe UI", 16, "bold"),
        text_color=["#7289da", "#7867c5"]
    )
    # 初始状态下显示提示
    overlay_frame.place(relx=0.5, rely=0.5, relwidth=0.4, relheight=0.2, anchor="center")
    overlay_text.place(relx=0.5, rely=0.5, anchor="center")
    
    # 创建预览区域的标签
    realtime_label = ctk.CTkLabel(
        right_frame,
        text="",
        fg_color=["#f0f0f4", "#1e1e2e"],
        corner_radius=10
    )
    realtime_label.place(relx=0.025, rely=0.02, relwidth=0.95, relheight=0.9)

    # 状态栏
    status_frame = ctk.CTkFrame(
        right_frame,
        corner_radius=8,
        fg_color=["#f0f0f4", "#292a37"],
        height=30
    )
    status_frame.place(relx=0.025, rely=0.93, relwidth=0.95, relheight=0.05)
    
    # 状态图标
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


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    preview.bind("<Up>", lambda event: update_face_reference(1))
    preview.bind("<Down>", lambda event: update_face_reference(-1))
    return preview


def update_status(text: str) -> None:
    global status_icon
    
    # 根据消息类型更新状态图标颜色
    if "错误" in text or "失败" in text:
        status_icon.configure(text_color=["#e74c3c", "#e74c3c"])
    elif "成功" in text or "完成" in text:
        status_icon.configure(text_color=["#2ecc71", "#2ecc71"])
    elif "警告" in text:
        status_icon.configure(text_color=["#f39c12", "#f39c12"])
    elif "请" in text:
        status_icon.configure(text_color=["#3498db", "#3498db"])
    else:
        status_icon.configure(text_color=["#7289da", "#7867c5"])
    
    status_label.configure(text=text)
    ROOT.update()


def select_source_path(source_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_SOURCE

    if source_path is None:
        source_path = ctk.filedialog.askopenfilename(
            title="select an source image",
            initialdir=RECENT_DIRECTORY_SOURCE,
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.webp"),),
        )
    if is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        image = render_image_preview(source_path, (200, 200))
        source_label.configure(image=image)
        if realtime_running:  # 如果实时换脸正在运行，通知状态已更新
            update_status("源图片已更新，实时换脸将使用新图片")
    else:
        roop.globals.source_path = None
        source_label.configure(image=None)


def select_target_path(target_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_TARGET

    if PREVIEW:
        PREVIEW.withdraw()
    clear_face_reference()
    if target_path is None:
        target_path = ctk.filedialog.askopenfilename(
            title="select an target image or video", initialdir=RECENT_DIRECTORY_TARGET
        )
    if is_image(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        image = render_image_preview(roop.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        roop.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT

    if is_image(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save image output file",
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save video output file",
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        roop.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(roop.globals.output_path)
        start()


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.contain(image, size, Image.LANCZOS)  # 等比缩放完整显示
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.contain(image, size, Image.LANCZOS)  # 等比缩放完整显示
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    global overlay_frame, overlay_text
    
    if PREVIEW.state() == "normal":
        PREVIEW.unbind("<Right>")
        PREVIEW.unbind("<Left>")
        PREVIEW.withdraw()
        clear_predictor()
        
        # 显示提示覆盖层
        if overlay_frame and overlay_text:
            overlay_frame.place(relx=0.5, rely=0.5, relwidth=0.4, relheight=0.2, anchor="center")
            overlay_text.place(relx=0.5, rely=0.5, anchor="center")
        
    elif roop.globals.source_path and roop.globals.target_path:
        init_preview()
        update_preview(roop.globals.reference_frame_number)
        PREVIEW.deiconify()


def init_preview() -> None:
    PREVIEW.title("Preview [ ↕ Reference face ]")
    if is_image(roop.globals.target_path):
        preview_slider.pack_forget()
    if is_video(roop.globals.target_path):
        video_frame_total = get_video_frame_total(roop.globals.target_path)
        if video_frame_total > 0:
            PREVIEW.title("Preview [ ↕ Reference face ] [ ↔ Frame number ]")
            PREVIEW.bind(
                "<Right>", lambda event: update_frame(int(video_frame_total / 20))
            )
            PREVIEW.bind(
                "<Left>", lambda event: update_frame(int(video_frame_total / -20))
            )
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(roop.globals.reference_frame_number)


def update_preview(frame_number: int = 0) -> None:
    global overlay_frame
    
    if roop.globals.source_path and roop.globals.target_path:
        # 隐藏提示覆盖层
        if overlay_frame:
            overlay_frame.place_forget()
            
        temp_frame = get_video_frame(roop.globals.target_path, frame_number)
        if predict_frame(temp_frame):
            sys.exit()
        source_face = get_one_face(cv2.imread(roop.globals.source_path))
        if not get_face_reference():
            reference_frame = get_video_frame(
                roop.globals.target_path, roop.globals.reference_frame_number
            )
            reference_face = get_one_face(
                reference_frame, roop.globals.reference_face_position
            )
            set_face_reference(reference_face)
        else:
            reference_face = get_face_reference()
        for frame_processor in get_frame_processors_modules(
            roop.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                source_face, reference_face, temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)


def update_face_reference(steps: int) -> None:
    clear_face_reference()
    reference_frame_number = int(preview_slider.get())
    roop.globals.reference_face_position += steps
    roop.globals.reference_frame_number = reference_frame_number
    update_preview(reference_frame_number)


def update_frame(steps: int) -> None:
    frame_number = preview_slider.get() + steps
    preview_slider.set(frame_number)
    update_preview(preview_slider.get())


def toggle_real_time_face_swap():
    global realtime_running, realtime_cap, overlay_frame
    if not realtime_running:
        start_real_time_face_swap()
    else:
        stop_real_time_face_swap()


def start_real_time_face_swap():
    global realtime_running, realtime_cap, overlay_frame

    if not roop.globals.source_path:
        update_status("请先选择源人脸图片")
        return

    source_face = get_one_face(cv2.imread(roop.globals.source_path))
    if not source_face:
        update_status("错误 - 源图片未检测到人脸")
        return

    realtime_cap = cv2.VideoCapture(0)
    if not realtime_cap.isOpened():
        update_status("错误 - 无法打开摄像头")
        return

    # 隐藏提示覆盖层
    if overlay_frame:
        overlay_frame.place_forget()
        
    realtime_running = True
    update_status("实时换脸中 - 点击按钮可停止")
    update_realtime_frame(source_face)


def stop_real_time_face_swap():
    global realtime_running, realtime_cap, overlay_frame, overlay_text
    
    realtime_running = False
    if realtime_cap:
        realtime_cap.release()
        realtime_cap = None
    
    # 清空图像但不显示文字
    realtime_label.configure(image=None, text="")
    
    # 显示提示覆盖层
    if overlay_frame and overlay_text:
        overlay_frame.place(relx=0.5, rely=0.5, relwidth=0.4, relheight=0.2, anchor="center")
        overlay_text.place(relx=0.5, rely=0.5, anchor="center")
    
    update_status("实时换脸已停止")


def update_realtime_frame(source_face):
    global realtime_running, realtime_cap
    if not realtime_running or not realtime_cap:
        return

    # 检查是否有新的源图片 - 如果有，重新获取源人脸
    if roop.globals.source_path:
        current_source_face = get_one_face(cv2.imread(roop.globals.source_path))
        if current_source_face is not None:
            source_face = current_source_face
            # 在UI中显示当前使用的图片名称
            current_image_name = os.path.basename(roop.globals.source_path)
            status_label.configure(text=f"当前使用: {current_image_name}")

    ret, frame = realtime_cap.read()
    if not ret:
        update_status("摄像头帧获取失败")
        stop_real_time_face_swap()
        return

    # 检测目标帧中的人脸并进行换脸
    target_face = get_one_face(frame)
    if target_face:
        for frame_processor in get_frame_processors_modules(
            roop.globals.frame_processors
        ):
            frame = frame_processor.process_frame(source_face, target_face, frame)

    # 转为tk可用图片并显示
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.contain(image, (800, 600), Image.LANCZOS)  # 修改预览画面大小
    tk_image = ctk.CTkImage(image, size=image.size)
    realtime_label.configure(image=tk_image)

    # 定时刷新
    realtime_label.after(30, lambda: update_realtime_frame(source_face))


def load_face_gallery(gallery_frame):
    """加载人脸库图片"""
    # 清除现有内容
    for widget in gallery_frame.winfo_children():
        widget.destroy()
    
    # 确保faces目录存在
    if not os.path.exists("faces"):
        os.makedirs("faces")
    
    # 获取所有图片
    face_images = glob.glob(os.path.join("faces", "*.png")) + glob.glob(
        os.path.join("faces", "*.jpg")
    ) + glob.glob(os.path.join("faces", "*.jpeg")) + glob.glob(
        os.path.join("faces", "*.webp")
    )
    
    # 如果没有图片，显示提示
    if not face_images:
        no_image_label = ctk.CTkLabel(gallery_frame, text="人脸库为空，请添加图片到faces文件夹")
        no_image_label.pack(expand=True, fill="both", padx=10, pady=10)
        return
    
    max_per_row = 4
    thumb_size = (48, 48)
    for idx, face_path in enumerate(face_images):
        try:
            img = Image.open(face_path)
            img = ImageOps.contain(img, thumb_size, Image.LANCZOS)  # 动态等比缩放
            tk_img = ctk.CTkImage(img, size=thumb_size)
            btn = ctk.CTkButton(
                gallery_frame,
                image=tk_img,
                text="",
                width=thumb_size[0],
                height=thumb_size[1],
                command=lambda p=face_path: select_source_path(p),
            )
            btn.grid(row=idx // max_per_row, column=idx % max_per_row, padx=6, pady=4)
        except Exception as e:
            print(f"加载图片错误: {face_path} - {str(e)}")
